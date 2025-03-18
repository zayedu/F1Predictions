#!/usr/bin/env python3
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def convert_time_to_seconds(time_val):
    """
    Converts a time string or pd.Timedelta (e.g., "0:01:26.123") to seconds.
    Returns NaN if conversion fails.
    """
    try:
        # If already a Timedelta type, extract total seconds.
        if isinstance(time_val, pd.Timedelta):
            return time_val.total_seconds()
        # Else assume string and use pd.to_timedelta.
        return pd.to_timedelta(time_val).total_seconds()
    except Exception as e:
        return np.nan

def load_and_combine_data():
    """
    Loads historical and current season CSV data and returns a combined DataFrame.
    Expects files "f1_2024_data.csv" and "f1_current_season_data.csv" in the working directory.
    """
    try:
        hist_df = pd.read_csv("f1_2024_data.csv")
        print(f"Historical data loaded: {len(hist_df)} rows.")
    except Exception as e:
        print("Error loading historical data:", e)
        hist_df = pd.DataFrame()

    try:
        curr_df = pd.read_csv("f1_current_season_data.csv")
        print(f"Current season data loaded: {len(curr_df)} rows.")
    except Exception as e:
        print("Error loading current season data:", e)
        curr_df = pd.DataFrame()

    if not hist_df.empty and not curr_df.empty:
        combined_df = pd.concat([hist_df, curr_df], ignore_index=True)
    elif not hist_df.empty:
        combined_df = hist_df.copy()
    elif not curr_df.empty:
        combined_df = curr_df.copy()
    else:
        combined_df = pd.DataFrame()

    print(f"Combined data has {len(combined_df)} rows.")
    return combined_df

def feature_engineering(df):
    """
    Performs feature engineering on the raw data:
      - Converts time columns (BestQualiLap and AvgSector times) to seconds.
      - Computes a DriverForm metric: average finishing position per driver per season.
      - Computes a TeamAvgPosition metric: average finishing position per team.
      - Incorporates weather: maps weather description to numeric codes.
      - One-hot encodes track/ event names.
      - Adds RoundNumber as a feature.
    Returns the enriched DataFrame.
    """
    # Convert time columns to seconds.
    df['BestQualiLap_s'] = df['BestQualiLap'].apply(convert_time_to_seconds)
    for col in ['AvgSector1', 'AvgSector2', 'AvgSector3']:
        new_col = col + "_s"
        df[new_col] = df[col].apply(convert_time_to_seconds)

    # Compute DriverForm: average finishing position per driver per season.
    driver_form = df.groupby(['Year', 'Abbreviation'])['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'DriverForm'}
    )
    df = pd.merge(df, driver_form, on=['Year', 'Abbreviation'], how='left')

    # Compute TeamAvgPosition: average finishing position per team.
    team_perf = df.groupby('RaceTeam')['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'TeamAvgPosition'}
    )
    df = pd.merge(df, team_perf, on='RaceTeam', how='left')

    # Process Weather information, if available.
    # Assume that if the column 'Weather' exists, it contains descriptive strings.
    if 'Weather' in df.columns:
        # Define a mapping: adjust as needed for your data.
        weather_mapping = {
            "Sunny": 0,
            "Clear": 0,
            "Cloudy": 1,
            "Overcast": 1,
            "Rain": 2,
            "Rainy": 2,
            "Wet": 3
        }
        df['Weather_code'] = df['Weather'].map(weather_mapping)
        # If any weather value wasn't mapped, fill with the median value.
        df['Weather_code'] = df['Weather_code'].fillna(df['Weather_code'].median())
    else:
        # If weather data is not available, use a default value.
        df['Weather_code'] = 0

    # One-hot encode track/event names (EventName)
    if 'EventName' in df.columns:
        event_dummies = pd.get_dummies(df['EventName'], prefix='Event')
        df = pd.concat([df, event_dummies], axis=1)

    # Ensure RoundNumber is numeric
    df['RoundNumber'] = pd.to_numeric(df['RoundNumber'], errors='coerce')

    # Drop rows missing any of the essential features or target.
    required_features = ['BestQualiLap_s', 'AvgSector1_s', 'AvgSector2_s', 'AvgSector3_s',
                         'DriverForm', 'TeamAvgPosition', 'Weather_code', 'RoundNumber']
    df = df.dropna(subset=required_features)

    print("Feature engineering complete. Sample data:")
    print(df.head())
    return df

def train_model(df):
    """
    Trains a Gradient Boosting Regressor to predict BestQualiLap_s using a range of features:
      - Average sector times in seconds.
      - DriverForm, TeamAvgPosition.
      - Weather_code, RoundNumber.
      - One-hot encoded EventName variables.
    Returns the trained model and prints the validation MAE.
    """
    # Define base features.
    feature_cols = ['AvgSector1_s', 'AvgSector2_s', 'AvgSector3_s', 'DriverForm', 'TeamAvgPosition', 'Weather_code', 'RoundNumber']

    # Add one-hot encoded event features if present.
    event_cols = [col for col in df.columns if col.startswith("Event_")]
    feature_cols.extend(event_cols)

    target_col = 'BestQualiLap_s'

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=402)

    # Train a Gradient Boosting Regressor.
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=402)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.3f} seconds")

    return model

def main():
    # 1. Load data from CSV files.
    df = load_and_combine_data()
    if df.empty:
        print("No data available for training. Exiting.")
        return

    # 2. Perform feature engineering.
    df = feature_engineering(df)

    # 3. Train the model.
    model = train_model(df)

    # 4. Save the trained model.
    model_filename = "qualifying_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved to {model_filename}")

if __name__ == "__main__":
    main()
