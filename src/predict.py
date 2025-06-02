import joblib
import pandas as pd

def predict_demand(model_path, station_id, date_time):
    # date_time as datetime object
    df = pd.DataFrame({'date_time': [pd.to_datetime(date_time)],
                       'station_id': [station_id]})
    # feature engineering
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    # Station dummies, ensure same columns as training
    for sid in [101,102,103,104,105]:
        df[f'station_{sid}'] = 1 if sid == station_id else 0
    features = ['hour', 'day_of_week', 'month'] + [f'station_{sid}' for sid in [101,102,103,104,105]]
    X = df[features]
    model = joblib.load(model_path)
    pred = model.predict(X)[0]
    return max(0, int(pred))