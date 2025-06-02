import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_processing import load_data, feature_engineering

def train_model(data_path, model_path):
    df = load_data(data_path)
    X, y = feature_engineering(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    # Optionally, return test data for evaluation
    return model, X_test, y_test

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    model, X_test, y_test = train_model(data_path, model_path)
    print("Model trained and saved to", model_path)