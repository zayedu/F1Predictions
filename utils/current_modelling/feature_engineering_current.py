#!/usr/bin/env python3
import pandas as pd
import numpy as np
from fastf1 import utils

def convert_time_to_seconds(time_val):
    try:
        return utils.delta_to_seconds(time_val)
    except Exception:
        try:
            return pd.to_timedelta(time_val).total_seconds()
        except Exception:
            return np.nan

def engineer_features(current_df, last_season_df=None):
    """
    Performs extended feature engineering using current season data and last season data (if available).
    Features include:
      - BestQualiLap_s: best qualifying lap time in seconds (target)
      - CurrentDriverForm: average finishing position from current season
      - LastDriverForm: average finishing position from last season (if available)
      - TeamAvgPosition: current season team performance
      - LastTeamPerf: last season team performance (if available)
      - LastTrackPerf: driver performance on the specific track from last season (if available)
      - Weather_code: numeric mapping of weather (from current season data)
      - One-hot encoded Event_ dummy columns.
      - RoundNumber
    """
    df = current_df.copy()

    # Convert BestQualiLap to seconds.
    df['BestQualiLap_s'] = df['BestQualiLap'].apply(convert_time_to_seconds)

    # Compute current driver form.
    current_form = df.groupby(['Year', 'Abbreviation'])['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'CurrentDriverForm'}
    )
    df = pd.merge(df, current_form, on=['Year', 'Abbreviation'], how='left')

    # Compute current team performance.
    current_team_perf = df.groupby('RaceTeam')['FinalPosition'].mean().reset_index().rename(
        columns={'FinalPosition': 'TeamAvgPosition'}
    )
    df = pd.merge(df, current_team_perf, on='RaceTeam', how='left')

    # Map Weather to numeric code.
    weather_mapping = {"Clear": 0, "Sunny": 0, "Cloudy": 1, "Overcast": 1, "Rain": 2, "Rainy": 2, "Wet": 3}
    if 'Weather' in df.columns:
        df['Weather_code'] = df['Weather'].map(weather_mapping).fillna(0).astype(int)
    else:
        df['Weather_code'] = 0

    # One-hot encode EventName.
    if 'EventName' in df.columns:
        df = pd.get_dummies(df, columns=['EventName'], prefix='Event')

    # Ensure RoundNumber is numeric.
    df['RoundNumber'] = pd.to_numeric(df['RoundNumber'], errors='coerce')

    # --- Incorporate Last Season Data (if available) ---
    if last_season_df is not None and not last_season_df.empty:
        ls_df = last_season_df.copy()
        ls_df['BestQualiLap_s'] = ls_df['BestQualiLap'].apply(convert_time_to_seconds)
        ls_form = ls_df.groupby(['Year', 'Abbreviation'])['FinalPosition'].mean().reset_index().rename(
            columns={'FinalPosition': 'LastDriverForm'}
        )
        ls_team = ls_df.groupby('RaceTeam')['FinalPosition'].mean().reset_index().rename(
            columns={'FinalPosition': 'LastTeamPerf'}
        )
        # For track-specific performance, if available.
        if 'EventName' in ls_df.columns:
            track_perf = ls_df.groupby(['Abbreviation', 'EventName'])['FinalPosition'].mean().reset_index().rename(
                columns={'FinalPosition': 'LastTrackPerf'}
            )
        else:
            track_perf = pd.DataFrame(columns=['Abbreviation', 'LastTrackPerf'])

        df = pd.merge(df, ls_form, on='Abbreviation', how='left')
        df = pd.merge(df, ls_team, on='RaceTeam', how='left')
        if not track_perf.empty:
            df = pd.merge(df, track_perf, on='Abbreviation', how='left')
    else:
        df['LastDriverForm'] = np.nan
        df['LastTeamPerf'] = np.nan
        df['LastTrackPerf'] = np.nan

    # Final schema.
    keep_cols = ['Abbreviation', 'Year', 'RoundNumber', 'BestQualiLap_s',
                 'CurrentDriverForm', 'LastDriverForm', 'TeamAvgPosition', 'LastTeamPerf', 'LastTrackPerf', 'Weather_code']
    event_cols = [col for col in df.columns if col.startswith("Event_")]
    keep_cols.extend(event_cols)
    return df[keep_cols]

if __name__=="__main__":
    current_df = pd.read_csv("f1_current_season_data.csv")
    try:
        last_df = pd.read_csv("f1_last_season_data.csv")
    except Exception:
        last_df = None
    features_df = engineer_features(current_df, last_season_df=last_df)
    print("Engineered extended features (first 5 rows):")
    print(features_df.head())
    features_df.to_csv("f1_current_season_features_extended.csv", index=False)
