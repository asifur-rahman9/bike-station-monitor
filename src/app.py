import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
from data_processing import load_data

# Constants
MODEL_PATH = "models/bike_demand_model.pkl"
DATA_PATH = "data/bixi_sample.csv"
STATION_IDS = [101, 102, 103, 104, 105]
FEATURES = ["hour", "day_of_week", "month"] + [f"station_{sid}" for sid in STATION_IDS]

# Load the trained model (cached so it only loads once)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.title("Montreal Bike-Share Demand Predictor 🚲📈")

# Sidebar: choose view mode and station
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Choose View Mode", ["Single Prediction", "Historical Trends", "Feature Importance"])
station = st.sidebar.selectbox("Station ID", STATION_IDS, index=0)

if mode == "Single Prediction":
    st.subheader("Predict Demand for a Specific Date & Time")
    # Date & time selection
    date_input = st.date_input("Select Date", pd.to_datetime("2021-01-01"))
    time_input = st.time_input("Select Time", pd.to_datetime("2021-01-01 08:00").time())
    dt = pd.to_datetime(f"{date_input} {time_input}")

    if st.button("Predict Demand"):
        # Build a single-row DataFrame with proper features
        df_point = pd.DataFrame({"date_time": [dt], "station_id": [station]})
        df_point["hour"] = df_point["date_time"].dt.hour
        df_point["day_of_week"] = df_point["date_time"].dt.dayofweek
        df_point["month"] = df_point["date_time"].dt.month
        for sid in STATION_IDS:
            df_point[f"station_{sid}"] = 1 if sid == station else 0

        X_point = df_point[FEATURES]
        pred = model.predict(X_point)[0]
        pred = max(0, int(pred))
        st.write(f"**Predicted trips at station {station} on {dt}: {pred}**")

elif mode == "Historical Trends":
    st.subheader("Compare Actual vs. Predicted Over a Date Range")
    # Load full dataset
    df = load_data(DATA_PATH)
    # Keep only rows for the selected station
    df_station = df[df["station_id"] == station].copy()

    # Let the user select a date range (between the earliest and latest date available)
    min_date = df_station["date_time"].min().date()
    max_date = df_station["date_time"].max().date()
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = date_range
        # Include all times on the end_date
        end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

        mask = (df_station["date_time"] >= pd.to_datetime(start_date)) & (df_station["date_time"] <= end_datetime)
        df_range = df_station.loc[mask].copy()

        if df_range.empty:
            st.write("No data available for this range.")
        else:
            # Feature engineering (vectorized) for the selected range
            df_range["hour"] = df_range["date_time"].dt.hour
            df_range["day_of_week"] = df_range["date_time"].dt.dayofweek
            df_range["month"] = df_range["date_time"].dt.month
            for sid in STATION_IDS:
                df_range[f"station_{sid}"] = 1 if sid == station else 0

            X_range = df_range[FEATURES]
            preds = model.predict(X_range)
            df_range["predicted_trips"] = preds.astype(int)
            df_range.sort_values("date_time", inplace=True)

            # Interactive line chart: actual vs predicted
            st.line_chart(df_range.set_index("date_time")[["trips", "predicted_trips"]])

            # Optionally show raw data in a table
            if st.checkbox("Show Raw Data Table"):
                st.dataframe(df_range[["date_time", "trips", "predicted_trips"]])

elif mode == "Feature Importance":
    st.subheader("Model Feature Importance")
    try:
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": FEATURES, "importance": importances})
            .sort_values(by="importance", ascending=False)
        )
        st.bar_chart(fi_df.set_index("feature")["importance"])

        if st.checkbox("Show Feature Importance Values"):
            st.dataframe(fi_df)
    except AttributeError:
        st.write("Feature importances not available for this model.")