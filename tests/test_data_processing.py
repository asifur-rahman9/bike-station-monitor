import pandas as pd
from datetime import datetime
from src.data_processing import load_data, feature_engineering
import os

def test_load_data(tmp_path):
    # Create a small CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'station_id': [101, 102],
        'date_time': ['2021-01-01 00:00:00', '2021-01-02 01:00:00'],
        'trips': [10, 15]
    })
    df.to_csv(csv_path, index=False)
    loaded = load_data(str(csv_path))
    assert isinstance(loaded, pd.DataFrame)
    assert 'date_time' in loaded.columns
    assert loaded.shape[0] == 2

def test_feature_engineering():
    df = pd.DataFrame({
        'station_id': [101, 102, 103],
        'date_time': pd.to_datetime(['2021-01-01 08:00:00', '2021-01-02 09:00:00', '2021-01-03 10:00:00']),
        'trips': [5, 7, 8]
    })
    X, y = feature_engineering(df)
    assert 'hour' in X.columns
    assert 'day_of_week' in X.columns
    assert 'month' in X.columns
    for sid in [101,102,103,104,105]:
        col = f'station_{sid}'
        assert col in X.columns
    assert len(y) == 3