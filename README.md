**Bike Share Demand Predictor**
===============================

A simple pipeline to train and serve a bike-share demand model with a Streamlit dashboard.

**Description**
---------------

This repository provides a full, end-to-end workflow for predicting bike-share demand:

1.  **Data Loading & Feature Engineering**
    
2.  **Model Training (regression) & Saving**
    
3.  **Single‐Row Prediction Logic**
    
4.  **Interactive Streamlit Dashboard**
    
5.  **Unit Tests for Data Processing & Prediction**
    

**Project Structure**
---------------------
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
```

**Features**
------------

### **Data Loading & Feature Engineering**

*   load\_data() reads a CSV into a pandas DataFrame.
    
*   feature\_engineering() creates time‐based features (hour, day, month, etc.) plus any station or weather metadata.
    

### **Model Training**

*   train.py trains a scikit‐learn regression model (e.g., RandomForestRegressor or LinearRegression).
    
*   Saves the trained model as models/bike\_demand\_model.pkl.
    

### **Single-Row Prediction**

*   predict.py loads the saved model and returns demand for a single record supplied as a dictionary or one‐row DataFrame.
    

### **Interactive Streamlit Dashboard**

*   app.py provides a UI to input date/time, weather, and station parameters to see real‐time demand estimates.
    
*   Accessible at **http://localhost:8501** once launched.
    

### **Automated Testing**

*   tests/test\_data\_processing.py validates data loading and engineered features.
    
*   tests/test\_predict.py checks that the prediction pipeline returns expected results.
    

**Requirements**
----------------

*   **Python 3.8+**
    
*   Core Python packages listed in requirements.txt (pandas, numpy, scikit‐learn, joblib, streamlit)
    

**Installation**
----------------

1.  **Clone the repository**
    
```bash 
git clone https://github.com/your-username/bike_share_project.git  cd bike_share_project
```

2.  **(Optional) Create & activate a virtual environment**
    
```bash
python3 -m venv venv  source venv/bin/activate      # macOS / Linux  # .\venv\Scripts\activate     # Windows   `
```

2.  **Install dependencies**

```bash
pip install --upgrade pip  pip install -r requirements.txt   `
```

**Usage**
---------

### **1 · Train the Model**

```bash
python3 src/train.py data/bixi_sample.csv models/bike_demand_model.pkl   `
``` 

*   *   data/bixi\_sample.csv – path to the CSV training data
        
    *   models/bike\_demand\_model.pkl – output path for the trained model artifact
        

### **2 · Run Unit Tests**

```bash
pytest   `
```

### **3 · Launch the Dashboard**

```bash
streamlit run src/app.py   `
```

*   Opens **http://localhost:8501** where you can interactively explore demand predictions.
    

**Example**
-----------

1.  **Train**
    

```bash
python3 src/train.py data/bixi_sample.csv models/bike_demand_model.pkl   `
```

2.  **Start Dashboard**
    
```bash
streamlit run src/app.py   `
```

2.  *   Select a Date/Time (e.g., 2025-06-01 08:00)
        
    *   Enter Temperature (°C), Humidity (%), Wind Speed (km/h), and Station ID
        
    *   Click **Predict** to view the forecasted bike demand
        

**Contributing**
----------------

1.  **Fork** the repository and create your feature branch:
    
```bash
git checkout -b feature/my-new-feature   `
```

2.  **Make changes** and add tests if applicable.
    
3.  **Commit** with a clear message:
    
```bash
git commit -m "Add my new feature"   `
```

2.  **Push** your branch and open a Pull Request.
    

**License**
-----------

Distributed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
