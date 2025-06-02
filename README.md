# Bike Share Demand Predictor

A simple pipeline to train and serve a bike-share demand model with a Streamlit dashboard.

---

## Description

This repository provides a full, end-to-end workflow for predicting bike-share demand:

1. **Data Loading & Feature Engineering**  
2. **Model Training (regression) & Artifact Saving**  
3. **Single-Row Prediction Logic**  
4. **Interactive Streamlit Dashboard**  
5. **Unit Tests for Data Processing & Prediction**  

---

## Project Structure

```bash
bike_share_project/
├─ data/                      # Sample CSV files (e.g., bixi_sample.csv)
├─ models/                    # Saved model artifacts (e.g., bike_demand_model.pkl)
├─ requirements.txt           # Python dependencies
├─ src/
│  ├─ data_processing.py      # load_data() and feature_engineering()
│  ├─ train.py                # Train & save a regression model
│  ├─ predict.py              # Load model & make single-row predictions
│  └─ app.py                  # Streamlit app for interactive forecasts
├─ tests/
│  ├─ test_data_processing.py # Unit tests for data processing
│  └─ test_predict.py         # Unit tests for prediction logic
└─ README.md                  # This file
