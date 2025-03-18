#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime
import joblib
import numpy as np
from feature_engineering_current import engineer_features

def get_next_race_info(year=2025):
    """
    Retrieves the FastF1 event schedule for the given year,
    converts EventDate to a date, and returns the next upcoming race.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    today = datetime.datetime.now().date()
    upcoming = schedule[schedule['EventDate'] >= today]
    if upcoming.empty:
        print("No upcoming races found.")
        return None
    upcoming = upcoming.sort_values('EventDate')
    return upcoming.iloc[0]

def get_current_driver_list(year=2025, round_number=None):
    """
    Retrieves the current driver list from the FP1 session.
    Falls back to the current season CSV if FP1 fails.
    Returns driver codes in uppercase.
    """
    try:
        fp1_session = fastf1.get_session(year, round_number, 'FP1')
        fp1_session.load()
        drivers = fp1_session.drivers
        if isinstance(drivers, dict):
            driver_list = list(drivers.keys())
        elif isinstance(drivers, list):
            driver_list = drivers
        else:
            driver_list = []
        if not driver_list:
            raise Exception("Empty driver list from FP1")
        return [drv.upper() for drv in driver_list]
    except Exception as e:
        print("Error loading current driver list from FP1:", e)
        try:
            df = pd.read_csv("f1_current_season_data.csv")
            return df["Abbreviation"].str.upper().unique().tolist()
        except Exception as e2:
            print("Error loading driver list from CSV:", e2)
            return []

def update_features_for_upcoming(df, next_race):
    """
    Updates the engineered features DataFrame for the upcoming race.
    Sets the RoundNumber to the upcoming race's round and ensures that
    the event dummy for the upcoming race is present and set to 1.
    """
    df = df.copy()
    df["RoundNumber"] = next_race["RoundNumber"]
    # Ensure event dummy column is present.
    event_dummy = "Event_" + str(next_race["EventName"]).replace(" ", "_")
    if event_dummy not in df.columns:
        df[event_dummy] = 0
    df[event_dummy] = 1
    return df

def predict_qualifying():
    """
    Predicts the qualifying ranking for the upcoming race using only current season data.
    If no Q session data exists (i.e. qualifying hasn’t occurred), it uses the overall
    season data (from the engineered features CSV) to predict lap times.
    """
    year = 2025
    next_race = get_next_race_info(year)
    if next_race is None:
        print("No upcoming race found.")
        return
    print(f"Next race: Round {next_race['RoundNumber']} - {next_race['EventName']} on {next_race['EventDate']}")

    # Get current driver list from FP1.
    current_driver_list = get_current_driver_list(year, next_race["RoundNumber"])
    if not current_driver_list:
        print("No current drivers found.")
        return

    # Load engineered features from completed races.
    try:
        features_df = pd.read_csv("f1_current_season_features.csv")
    except Exception as e:
        print("Error loading engineered features:", e)
        return
    features_df["Abbreviation"] = features_df["Abbreviation"].str.upper()
    # Filter to include only current drivers.
    features_df = features_df[features_df["Abbreviation"].isin(current_driver_list)]
    if features_df.empty:
        print("No matching drivers found in engineered features.")
        return

    # Update features for the upcoming race.
    features_df = update_features_for_upcoming(features_df, next_race)

    # Define feature columns (all except Abbreviation, Year, and BestQualiLap_s).
    exclude_cols = ['Abbreviation', 'Year', 'BestQualiLap_s']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X_new = features_df[feature_cols]

    # Load the trained qualifying model.
    try:
        model = joblib.load("current_qualifying_model.pkl")
        print("Qualifying model loaded.")
    except Exception as e:
        print("Error loading qualifying model:", e)
        return

    # Reindex to ensure the features match those used in training.
    expected_features = model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    predictions = model.predict(X_new)
    features_df["PredictedQualiLap_s"] = predictions
    features_df = features_df.sort_values("PredictedQualiLap_s")
    features_df["PredictedQualiRank"] = range(1, len(features_df) + 1)

    # Filter final predictions to only include current drivers.
    final_df = features_df[features_df["Abbreviation"].isin(current_driver_list)]
    print("\nPredicted Qualifying Ranking:")
    print(final_df[["Abbreviation", "PredictedQualiLap_s", "PredictedQualiRank"]])
    return final_df

if __name__ == "__main__":
    predict_qualifying()
