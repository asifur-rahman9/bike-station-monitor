import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date_time'])
    return df

def feature_engineering(df):
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    # One-hot encode station_id for fixed list
    station_ids = [101,102,103,104,105]
    station_dummies = pd.get_dummies(df['station_id'], prefix='station')
    for sid in station_ids:
        col = f'station_{sid}'
        if col not in station_dummies.columns:
            station_dummies[col] = 0
    station_dummies = station_dummies[[f'station_{sid}' for sid in station_ids]]
    df = pd.concat([df, station_dummies], axis=1)
    features = ['hour', 'day_of_week', 'month'] + [f'station_{sid}' for sid in station_ids]
    X = df[features]
    y = df['trips']
    return X, y