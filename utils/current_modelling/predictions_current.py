#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime
import joblib
import numpy as np
from feature_engineering_extended import engineer_features

def get_next_race_info(year=2025):
    """
    Retrieves the next upcoming race from the FastF1 event schedule.
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
    Falls back to current season CSV if needed.
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
    Sets the RoundNumber to the upcoming race's round and ensures that the event dummy is present.
    """
    df = df.copy()
    df["RoundNumber"] = next_race["RoundNumber"]
    event_dummy = "Event_" + str(next_race["EventName"]).replace(" ", "_")
    if event_dummy not in df.columns:
        df[event_dummy] = 0
    df[event_dummy] = 1
    return df

def predict_qualifying():
    """
    Predicts the qualifying ranking for the upcoming race using the extended features.
    If qualifying session (Q) data is not available, it falls back to using overall season data.
    """
    year = 2025
    next_race = get_next_race_info(year)
    if next_race is None:
        print("No upcoming race found.")
        return
    print(f"Next race: Round {next_race['RoundNumber']} - {next_race['EventName']} on {next_race['EventDate']}")

    current_driver_list = get_current_driver_list(year, next_race["RoundNumber"])
    if not current_driver_list:
        print("No current drivers found.")
        return

    try:
        current_df = pd.read_csv("f1_current_season_data.csv")
    except Exception as e:
        print("Error loading current season data:", e)
        return

    # Engineer extended features using current season and (if available) last season data.
    try:
        last_df = pd.read_csv("f1_last_season_data.csv")
    except Exception:
        last_df = None

    features_df = engineer_features(current_df, last_season_df=last_df)
    features_df["Abbreviation"] = features_df["Abbreviation"].str.upper()
    features_df = features_df[features_df["Abbreviation"].isin(current_driver_list)]
    if features_df.empty:
        print("No matching drivers found in engineered features.")
        return

    features_df = update_features_for_upcoming(features_df, next_race)

    exclude_cols = ['Abbreviation', 'Year', 'BestQualiLap_s']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X_new = features_df[feature_cols]

    try:
        model = joblib.load("current_qualifying_model.pkl")
        print("Qualifying model loaded.")
    except Exception as e:
        print("Error loading qualifying model:", e)
        return

    expected_features = model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    predictions = model.predict(X_new)
    features_df["PredictedQualiLap_s"] = predictions
    features_df = features_df.sort_values("PredictedQualiLap_s")
    features_df["PredictedQualiRank"] = range(1, len(features_df)+1)

    final_df = features_df[features_df["Abbreviation"].isin(current_driver_list)]
    print("\nPredicted Qualifying Ranking:")
    print(final_df[["Abbreviation", "PredictedQualiLap_s", "PredictedQualiRank"]])
    return final_df

if __name__ == "__main__":
    predict_qualifying()
