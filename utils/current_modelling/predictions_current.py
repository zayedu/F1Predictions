#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime
import joblib
import numpy as np
from feature_engineering_current import engineer_features, create_best_quali_time, process_fp3_session

# Hard-coded current driver grid (team : list of driver full names)
F1_GRID_2025 = {
    "Red Bull Racing": ["Max Verstappen", "Liam Lawson"],
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "Mercedes": ["Andrea Kimi Antonelli", "George Russell"],
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Alpine": ["Pierre Gasly", "Jack Doohan"],
    "Williams": ["Alexander Albon", "Carlos Sainz Jr."],
    "Racing Bulls": ["Isack Hadjar", "Yuki Tsunoda"],
    "Haas": ["Oliver Bearman", "Esteban Ocon"],
    "Sauber": ["Nico Hülkenberg", "Gabriel Bortoleto"]
}

# Mapping from full driver name to driver abbreviation.
DRIVER_ABBREV = {
    "Max Verstappen": "VER",
    "Liam Lawson": "LAW",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton": "HAM",
    "Andrea Kimi Antonelli": "ANT",
    "George Russell": "RUS",
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Pierre Gasly": "GAS",
    "Jack Doohan": "DOO",
    "Alexander Albon": "ALB",
    "Carlos Sainz Jr.": "SAI",
    "Isack Hadjar": "HAD",
    "Yuki Tsunoda": "TSU",
    "Oliver Bearman": "BEA",
    "Esteban Ocon": "OCO",
    "Nico Hülkenberg": "HUL",
    "Gabriel Bortoleto": "BOR"
}

# Mapping from team name to a short team abbreviation.
TEAM_ABBREV = {
    "Red Bull Racing": "RBR",
    "Ferrari": "FER",
    "Mercedes": "MER",
    "McLaren": "MCL",
    "Aston Martin": "AST",
    "Alpine": "ALP",
    "Williams": "WIL",
    "Racing Bulls": "RB",
    "Haas": "HAS",
    "Sauber": "SAU"
}

# Build a dictionary mapping driver abbreviation to team name based on F1_GRID_2025.
def build_driver_to_team(grid):
    mapping = {}
    for team, drivers in grid.items():
        for full_name in drivers:
            abbrev = DRIVER_ABBREV.get(full_name)
            if abbrev:
                mapping[abbrev] = team
    return mapping

DRIVER_TO_TEAM = build_driver_to_team(F1_GRID_2025)

def get_current_driver_list(year=2025, round_number=None):
    """
    Returns the current driver list as a list of abbreviations based on the hard-coded grid.
    """
    driver_list = []
    for team, drivers in F1_GRID_2025.items():
        for full_name in drivers:
            abbrev = DRIVER_ABBREV.get(full_name)
            if abbrev:
                driver_list.append(abbrev)
    return driver_list

def get_next_race_info(year=2025):
    """
    Retrieves the next upcoming race from FastF1's schedule.
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

def update_features_for_upcoming(df, next_race):
    """
    Updates the DataFrame for the upcoming race by setting the correct RoundNumber and
    ensuring that the event dummy column exists and is set to 1.
    """
    df = df.copy()
    df["RoundNumber"] = next_race["RoundNumber"]
    event_dummy = "Event_" + str(next_race["EventName"]).replace(" ", "_")
    if event_dummy not in df.columns:
        df[event_dummy] = 0
    df[event_dummy] = 1
    return df

def predict_next_qualifying(next_race):
    """
    Uses historical combined current season data (and FP3 data, if available)
    to predict qualifying lap times for the upcoming race.
    """
    current_csv = "f1_current_season_combined.csv"
    try:
        features_df = engineer_features(current_csv, last_season_csv=None)
    except Exception as e:
        print("Error during extended feature engineering:", e)
        return None

    # Attempt to update features with FP3 data for the upcoming race.
    try:
        fp3_df = process_fp3_session(2025, next_race["RoundNumber"], next_race["EventName"])
        if not fp3_df.empty:
            fp3_df["Abbreviation"] = fp3_df["Abbreviation"].str.upper()
            for idx, row in fp3_df.iterrows():
                driver = row["Abbreviation"]
                mask = features_df["Abbreviation"] == driver
                features_df.loc[mask, "FP3_BestQualiLap"] = row.get("FP3_BestQualiLap")
            features_df["BestQualiLap_s"] = features_df.apply(create_best_quali_time, axis=1)
    except Exception as e:
        print("Error retrieving FP3 data for next race:", e)

    # Update features for the upcoming race.
    features_df = update_features_for_upcoming(features_df, next_race)
    current_driver_list = get_current_driver_list(2025, next_race["RoundNumber"])
    features_df = features_df[features_df["Abbreviation"].isin(current_driver_list)]
    if features_df.empty:
        print("No matching drivers found in engineered features.")
        return None

    # Aggregate duplicate rows so each driver appears only once.
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in features_df.columns if col not in numeric_cols and col != "Abbreviation"]
    aggregated = features_df.groupby("Abbreviation").agg({col: "mean" for col in numeric_cols}).reset_index()
    first_rows = features_df.drop_duplicates("Abbreviation")[["Abbreviation"] + non_numeric_cols]
    features_df = pd.merge(aggregated, first_rows, on="Abbreviation", how="left")

    # Now, add TeamName and TeamAbbrev columns using our mapping.
    features_df["TeamName"] = features_df["Abbreviation"].apply(lambda x: DRIVER_TO_TEAM.get(x, "Unknown"))
    features_df["TeamAbbrev"] = features_df["TeamName"].apply(lambda x: TEAM_ABBREV.get(x, "UNK"))

    exclude_cols = ['Abbreviation', 'Year', 'BestQualiLap_s']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X_new = features_df[feature_cols]

    try:
        model = joblib.load("current_qualifying_model.pkl")
        print("Qualifying model loaded.")
    except Exception as e:
        print("Error loading qualifying model:", e)
        return None

    expected_features = model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    predictions = model.predict(X_new)
    features_df["PredictedQualiLap_s"] = predictions
    features_df = features_df.sort_values("PredictedQualiLap_s")
    features_df["PredictedQualiRank"] = range(1, len(features_df) + 1)
    print("\nPredicted Qualifying Ranking (using FP3 + historical data):")
    print(features_df[["Abbreviation", "TeamName", "TeamAbbrev", "PredictedQualiLap_s", "PredictedQualiRank"]])
    return features_df

def predict_race_positions(next_race):
    """
    Placeholder: When qualifying session data is available, merge actual Q data with historical features
    to predict race positions.
    """
    print("Qualifying session has occurred; race prediction logic to be implemented.")
    return None

def predict_qualifying_or_race():
    year = 2025
    next_race = get_next_race_info(year)
    if next_race is None:
        print("No upcoming race found.")
        return None
    print(f"Next race: Round {next_race['RoundNumber']} - {next_race['EventName']} on {next_race['EventDate']}")

    qualifying_occurred = False
    try:
        q_session = fastf1.get_session(year, next_race["RoundNumber"], 'Q')
        q_session.load(laps=True)
        if not q_session.laps.empty:
            qualifying_occurred = True
    except Exception as e:
        print("Qualifying session not available:", e)
        qualifying_occurred = False

    if not qualifying_occurred:
        return predict_next_qualifying(next_race)
    else:
        return predict_race_positions(next_race)

if __name__ == "__main__":
    predict_qualifying_or_race()
