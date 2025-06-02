# Bike Share Demand Predictor

A simple pipeline to train and serve a bike‐share demand model with a Streamlit dashboard.

## Project Structure

bike_share_project/
├─ data/                      # Sample CSV files (e.g., bixi_sample.csv)
├─ models/                    # Saved model artifacts (e.g., bike_demand_model.pkl)
├─ requirements.txt           # Python dependencies
├─ src/
│  ├─ data_processing.py      # load_data() and feature_engineering()
│  ├─ train.py                # Train & save a regression model
│  ├─ predict.py              # Load model & make single‐row predictions
│  └─ app.py                  # Streamlit app for interactive forecasts
├─ tests/
│  ├─ test_data_processing.py # Unit tests for data processing
│  └─ test_predict.py         # Unit tests for prediction logic
└─ README.md                  # This file


## Features

- **Data Loading & Feature Engineering**  
  - `load_data()` reads a CSV into a DataFrame.  
  - `feature_engineering()` creates time‐based features.

- **Model Training**  
  - `train.py` trains a scikit‐learn regression (e.g., RandomForest or LinearRegression)  
  - Saves the fitted model as `models/bike_demand_model.pkl`.

- **Prediction Endpoint**  
  - `predict.py` loads the saved model and returns a predicted demand for a single record.

- **Interactive Dashboard**  
  - `app.py` (Streamlit) allows you to input date, weather, and location parameters to see demand estimates.

- **Automated Testing**  
  - `tests/test_data_processing.py` verifies data loading and feature outputs.  
  - `tests/test_predict.py` checks that the prediction pipeline returns expected shapes/values.

## Requirements

- Python 3.8+  
- See `requirements.txt` (core packages): pandas, numpy, scikit-learn, joblib, streamlit
