# Doctor Recommendation Model

This repository contains the implementation of the **Doctor Recommendation Model**, a system designed to suggest doctors based on users' health issues and their location. The project leverages machine learning techniques, particularly PyTorch, to deliver personalized and accurate recommendations.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Finding the right doctor for specific health concerns can be challenging. This project aims to simplify the process by providing recommendations tailored to a user's symptoms and proximity to medical professionals. The system is intended to:

1. Understand user health issues.
2. Provide location-based doctor suggestions.
3. Offer a seamless user experience through a web interface.

---

## Features

- **Symptom-Based Matching:** Recommends doctors based on health symptoms entered by the user.
- **Location Awareness:** Filters recommendations by the user's location for convenience.
- **Machine Learning-Powered:** Utilizes a PyTorch-based recommendation model for accuracy.
- **Web Integration:** Includes a front-end interface for easy user interaction.

---

## Project Structure

```
DoctorRecommendationModel/
├── backend/
│   ├── main.py                # Backend application entry point
│   ├── MODEL_WEIGHTS.pth      # Pre-trained model weights
├── Nepal_Doctor_Records.csv   # Dataset containing doctor details
├── Symptoms_Specialization.csv # Dataset mapping symptoms to specializations
├── DoctorRecommendationModel.ipynb # Jupyter notebook for model training
├── frontendside/              # Frontend application files
├── node_modules/              # Node.js dependencies
├── package.json               # Frontend configuration file
├── venv/                      # Python virtual environment
├── .gitignore                 # Git ignore file
├── .gitattributes             # Git attributes file
```

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/DoctorRecommendationModel.git
   cd DoctorRecommendationModel
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Backend Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Frontend Dependencies:**
   ```bash
   cd frontendside
   npm install
   ```

5. **Run the Application:**
   - Start the backend:
     ```bash
     python backend/main.py
     ```
   - Start the frontend:
     ```bash
     cd frontendside
     npm start
     ```

---

## Usage

1. Navigate to the frontend interface by opening your browser at `http://localhost:3000` (or the specified port).
2. Enter your symptoms and location in the provided fields.
3. View the list of recommended doctors based on your input.

---

## Dataset

The dataset used for training includes:
- **Doctor Details:** Specializations, location, availability.
- **Symptom Mappings:** Common health issues and associated medical fields.

For privacy and compliance, the dataset is anonymized and adheres to applicable data protection regulations.

---

## Model Architecture

The recommendation system uses:
- **Embedding Layers:** For encoding symptoms and doctor profiles.
- **Multi-Layer Perceptron (MLP):** For mapping symptoms to doctor recommendations.
- **Location Filtering:** Post-model filtering for location-based suggestions.

The model is built using PyTorch and optimized for performance.

---

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** React.js, Node.js
- **Machine Learning:** PyTorch
- **Database:** SQLite/PostgreSQL (as applicable)

---

## Future Improvements

- **Expanded Dataset:** Incorporate a larger dataset for improved recommendations.
- **Natural Language Processing (NLP):** Allow users to enter symptoms in plain text for better usability.
- **Mobile App Integration:** Provide a mobile-friendly version of the application.
- **Advanced Filters:** Add options for insurance, consultation fees, and availability.

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
