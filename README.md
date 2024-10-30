# Speech Therapy App - Detecting Pronunciation Differences

## Overview

This project aims to develop a machine learning model using ML to differentiate between the correctly and incorrectly pronounced words. The ultimate goal is to create a speech therapy app that helps children with speech disorders practice and improve their pronunciation skills independently, without requiring parental / speech therapist assistance.

## Project Components

1. **Audio Feature Extraction**:
   - **File**: `preprocess.py`
   - **Description**: Extracts relevant audio features from recordings and saves them into a CSV file for model training.

2. **Model Training**:
   - **File**: `model.py`
   - **Description**: Trains a Random Forest model using the features extracted and evaluates its performance.

3. **Model Evaluation**:
   - **File**: `testing.py`
   - **Description**: Evaluates the trained model on new data and outputs predictions.

4. **Frontend Integration**:
   - **File**: TBU
   - **Description**: The model will be integrated with a React frontend to provide an interactive platform where users can test and improve their pronunciation.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `soundfile`
- `joblib`
- `librosa`
- `pydub`

You can install them using:
```bash
pip install -r requirements.txt
