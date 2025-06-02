import pandas as pd
import joblib
import os
from datetime import datetime
from src.predict import predict_demand

def test_predict_demand(tmp_path):
    # Create a dummy model that always predicts 42
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    dummy_model = RandomForestRegressor(n_estimators=1, random_state=42)
    # Train on dummy data
    X = pd.DataFrame({
        'hour': [0,1],
        'day_of_week': [0,1],
        'month': [1,1],
        'station_101': [1,0],
        'station_102': [0,1],
        'station_103': [0,0],
        'station_104': [0,0],
        'station_105': [0,0]
    })
    y = np.array([10,20])
    dummy_model.fit(X, y)
    model_path = tmp_path / "dummy_model.pkl"
    joblib.dump(dummy_model, model_path)

    # Test prediction
    pred = predict_demand(str(model_path), 101, datetime(2021,1,1,8,0))
    assert isinstance(pred, int)