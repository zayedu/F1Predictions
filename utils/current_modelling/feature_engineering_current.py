#!/usr/bin/env python3
import pandas as pd
import numpy as np
from fastf1 import utils

def convert_time_to_seconds(time_val):
    try:
        return utils.delta_to_seconds(time_val)
    except Exception:
        # Fallback: use pandas to_timedelta
        try:
            return pd.to_timedelta(time_val).total_seconds()
        except Exception:
            return np.nan

def engineer_features(df):
    """
    Performs feature engineering on current season data.
    Produces the following columns:
      - BestQualiLap_s: the best qualifying lap time in seconds (target).
      - DriverForm: average finishing position per driver.
      - TeamAvgPosition: average finishing position per team.
      - Weather_code: a placeholder (0 by default).
      - One-hot encoded EventName columns.
      - RoundNumber (numeric).
    Returns a DataFrame with a standardized schema.
    """
    # Convert BestQualiLap to seconds (target)
    df['BestQualiLap_s'] = df['BestQualiLap'].apply(convert_time_to_seconds)

    # Compute DriverForm from FinalPosition per driver.
    driver_form = df.groupby(['Year', 'Abbreviation'])['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'DriverForm'}
    )
    df = pd.merge(df, driver_form, on=['Year', 'Abbreviation'], how='left')

    # Compute TeamAvgPosition per team.
    team_perf = df.groupby('RaceTeam')['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'TeamAvgPosition'}
    )
    df = pd.merge(df, team_perf, on='RaceTeam', how='left')

    # Set Weather_code. (If no weather column, default to 0.)
    if 'Weather' in df.columns:
        weather_mapping = {"Clear": 0, "Sunny": 0, "Cloudy": 1, "Overcast": 1, "Rain": 2, "Rainy": 2, "Wet": 3}
        df['Weather_code'] = df['Weather'].map(weather_mapping).fillna(0).astype(int)
    else:
        df['Weather_code'] = 0

    # One-hot encode EventName.
    if 'EventName' in df.columns:
        df = pd.get_dummies(df, columns=['EventName'], prefix='Event')

    # Ensure RoundNumber is numeric.
    df['RoundNumber'] = pd.to_numeric(df['RoundNumber'], errors='coerce')

    # Keep only needed columns.
    # We'll keep: Abbreviation, Year, RoundNumber, BestQualiLap_s, DriverForm, TeamAvgPosition, Weather_code, plus event dummy columns.
    cols = ['Abbreviation', 'Year', 'RoundNumber', 'BestQualiLap_s', 'DriverForm', 'TeamAvgPosition', 'Weather_code']
    event_cols = [col for col in df.columns if col.startswith("Event_")]
    cols.extend(event_cols)
    return df[cols]

if __name__=="__main__":
    df = pd.read_csv("f1_current_season_data.csv")
    features_df = engineer_features(df)
    print("Engineered features (first 5 rows):")
    print(features_df.head())
    features_df.to_csv("f1_current_season_features.csv", index=False)
