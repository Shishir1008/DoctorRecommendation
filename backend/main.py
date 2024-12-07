import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load datasets
try:
    symptoms_df = pd.read_csv(
        r'C:\Users\shish\OneDrive\Desktop\DoctorRecommendationApp\backend\Symptoms_Speicalization.csv'
    )
    doctors_df = pd.read_csv(
        r'C:\Users\shish\OneDrive\Desktop\DoctorRecommendationApp\backend\Nepal_Doctor_Recommendations.csv'
    )
except FileNotFoundError as e:
    raise RuntimeError(f"CSV file not found: {e}")

# Extract unique locations and specializations
locations = doctors_df['Location'].dropna().unique().tolist()
specializations = symptoms_df['Specialization'].dropna().unique().tolist()

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class SymptomModel(nn.Module):
    def __init__(self):
        super(SymptomModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.specialization_classifier = nn.Linear(self.distilbert.config.hidden_size, len(specializations))

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token output
        logits = self.specialization_classifier(pooled_output)
        return logits

# Load the pre-trained model
model = SymptomModel()
try:
    model.load_state_dict(torch.load(r'C:\Users\shish\OneDrive\Desktop\DoctorRecommendationApp\backend\MODEL_WEIGHTS.pth', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError as e:
    raise RuntimeError(f"Model weights file not found: {e}")

# Define the input schema
class SymptomInput(BaseModel):
    symptom: str
    location: str

@app.get("/")
def root():
    """
    Root endpoint to check service health.
    """
    return {"message": "Welcome to the Doctor Recommendation System!"}

@app.get("/locations/")
def get_locations():
    """
    Returns available locations.
    """
    return {"locations": locations}


@app.post("/predict/")
async def predict_specialization(input: SymptomInput):
    """
    Predicts the specialization based on the symptom and filters doctors by location.
    """
    try:
        # Validate location
        if input.location not in locations:
            raise HTTPException(status_code=400, detail="Invalid location selected.")

        # Tokenize the symptom for model input
        inputs = tokenizer.encode_plus(
            input.symptom,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Perform model inference
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            pred_index = torch.argmax(logits, dim=1).item()

        specialization = specializations[pred_index]

        # Filter doctors by specialization and location
        recommended_doctors = doctors_df[
            (doctors_df['Specialization'] == specialization) & 
            (doctors_df['Location'] == input.location)
        ]

        # Return response
        if recommended_doctors.empty:
            return {
                "specialization": specialization,
                "message": "No doctors found for the selected specialization and location."
            }

        # Convert filtered DataFrame to a list of dictionaries
        doctors_list = recommended_doctors[[
            'Doctor Name', 'Specialization', 'Location', 'Phone Number', 'Qualification', 'Ratings'
        ]].to_dict(orient="records")

        return {
            "specialization": specialization,
            "doctors": doctors_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
