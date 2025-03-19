#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_qualifying_model(df):
    """
    Trains a Gradient Boosting Regressor to predict the qualifying lap time (in seconds)
    using the extended features.

    Expected columns include:
      - BestQualiLap_s (target)
      - CurrentDriverForm, LastDriverForm, TeamAvgPosition, LastTeamPerf, LastTrackPerf,
        RoundNumber, Weather_code, and one-hot encoded Event_ columns.

    Non-feature columns like 'Abbreviation' and 'Year' are excluded.
    Missing values are imputed with the column mean.
    """
    # Define which columns to exclude from features.
    exclude_cols = ['Abbreviation', 'Year', 'BestQualiLap_s']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target = 'BestQualiLap_s'

    # Impute missing values for each feature and the target.
    for col in feature_cols + [target]:
        mean_val = df[col].mean()
        # Fill missing values with mean if computed; otherwise, use 0.
        df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0)

    # Drop any remaining rows with missing values.
    df = df.dropna(subset=feature_cols + [target])
    if df.empty:
        raise ValueError("No data left after imputation.")

    # Separate features and target.
    X = df[feature_cols]
    y = df[target]

    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting Regressor.
    model = HistGradientBoostingRegressor( learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set.
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.3f} seconds")
    return model

if __name__=="__main__":
    # Load the extended feature file produced by your feature engineering module.
    df = pd.read_csv("f1_current_season_features_extended.csv")
    model = train_qualifying_model(df)
    joblib.dump(model, "current_qualifying_model.pkl")
    print("Model saved as current_qualifying_model.pkl")
