"""Streamlit demo (Person A): predict hourly bike rentals.
Run from the project root:  python -m streamlit run demo/app.py
The model (SVR) is trained from data/train.csv the first time the app opens and then cached.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
NUM = ["temp", "atemp", "humidity", "windspeed", "workingday", "year",
       "peak_weekday", "peak_weekend"]
CATS = {"hour": range(24), "season": range(1, 5), "weather": range(1, 5)}
WEATHER = {1: "Clear / partly cloudy", 2: "Mist / cloudy", 3: "Light rain / snow",
           4: "Heavy rain / snow"}


def make_features(df):
    """Same ideas as the project: hour, year, peak-hour flags, weather; holiday dropped."""
    dt = pd.to_datetime(df["datetime"])
    X = pd.DataFrame(index=df.index)
    X["hour"] = dt.dt.hour
    X["year"] = dt.dt.year - 2011
    for c in ["season", "weather", "temp", "atemp", "humidity", "windspeed", "workingday"]:
        X[c] = df[c].values
    wd = X["workingday"] == 1
    X["peak_weekday"] = (wd & X["hour"].isin([7, 8, 9, 17, 18, 19])).astype(int)
    X["peak_weekend"] = ((~wd) & X["hour"].between(10, 18)).astype(int)
    for c, cats in CATS.items():                      # fixed categories -> same columns always
        X[c] = pd.Categorical(X[c], categories=list(cats))
    return pd.get_dummies(X, columns=list(CATS), dtype=float)


def train_model(path=DATA_PATH):
    train = pd.read_csv(path)
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
    model.fit(make_features(train), np.log1p(train["count"]))
    return model


def predict(model, df):
    return np.clip(np.expm1(model.predict(make_features(df))), 0, None)


@st.cache_resource(show_spinner="Training model on train.csv (first run only)...")
def load_model():
    return train_model()


def main():
    st.set_page_config(page_title="Bike Demand Predictor", page_icon="🚲")
    st.title("🚲 Bike Rental Demand Predictor")
    st.caption("Capital Bikeshare, Washington D.C. | SVR (RBF) on log(count)")
    model = load_model()

    c1, c2 = st.columns(2)
    with c1:
        date = st.date_input("Date", value=pd.Timestamp(2012, 6, 15),
                             min_value=pd.Timestamp(2011, 1, 1), max_value=pd.Timestamp(2012, 12, 31))
        hour = st.slider("Hour of day", 0, 23, 8)
        holiday = st.checkbox("Public holiday")
        weather = st.selectbox("Weather", list(WEATHER), format_func=lambda k: WEATHER[k])
    with c2:
        temp = st.slider("Temperature (°C)", 0.0, 41.0, 22.0)
        atemp = st.slider("'Feels like' temperature (°C)", 0.0, 45.0, 24.0)
        humidity = st.slider("Humidity (%)", 0, 100, 55)
        windspeed = st.slider("Wind speed", 0.0, 57.0, 10.0)

    day = pd.Timestamp(date)
    workingday = int(day.dayofweek < 5 and not holiday)
    season = (day.month - 1) // 3 + 1                  # dataset convention: 1 = Jan-Mar

    rows = pd.DataFrame({
        "datetime": [day + pd.Timedelta(hours=h) for h in range(24)],
        "season": season, "workingday": workingday, "weather": weather,
        "temp": temp, "atemp": atemp, "humidity": humidity, "windspeed": windspeed,
    })
    preds = predict(model, rows)

    st.metric(f"Predicted rentals at {hour:02d}:00", f"{preds[hour]:.0f} bikes")
    st.subheader("Predicted rentals across the day (same conditions)")
    st.line_chart(pd.DataFrame({"predicted rentals": preds}, index=range(24)))


if __name__ == "__main__":
    main()