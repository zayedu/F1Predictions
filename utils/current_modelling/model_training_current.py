#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

def train_qualifying_model(df):
    """
    Trains a Gradient Boosting Regressor to predict BestQualiLap_s using extended features.
    Uses features: CurrentDriverForm, LastDriverForm, TeamAvgPosition, LastTeamPerf, LastTrackPerf,
    RoundNumber, Weather_code, and event dummies.
    Missing values are imputed with column means.
    """
    exclude_cols = ['Abbreviation', 'Year', 'BestQualiLap_s']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target = 'BestQualiLap_s'

    for col in feature_cols + [target]:
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(mean_val)

    df = df.dropna(subset=feature_cols + [target])
    if df.empty:
        raise ValueError("No data left after imputation.")

    X = df[feature_cols]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.3f} seconds")
    return model

if __name__=="__main__":
    df = pd.read_csv("f1_current_season_features_extended.csv")
    model = train_qualifying_model(df)
    joblib.dump(model, "current_qualifying_model.pkl")
    print("Model saved as current_qualifying_model.pkl")
