#!/usr/bin/env python3
import pandas as pd
import numpy as np
from fastf1 import utils

def convert_time_to_seconds(time_val):
    try:
        return utils.delta_to_seconds(time_val)
    except Exception:
        return np.nan

def engineer_features(df):
    """
    Performs feature engineering using only current season data.
    - Converts lap times to seconds.
    - Computes driver form as the average finishing position for each driver.
    - Computes team performance as the average finishing position for each team.
    - One-hot encodes the EventName.
    """
    # Convert BestQualiLap to seconds.
    df['BestQualiLap_s'] = df['BestQualiLap'].apply(convert_time_to_seconds)
    # Compute DriverForm.
    driver_form = df.groupby(['Year', 'Abbreviation'])['FinalPosition'].mean().reset_index().rename(columns={'FinalPosition':'DriverForm'})
    df = pd.merge(df, driver_form, on=['Year','Abbreviation'], how='left')
    # Compute TeamAvgPosition.
    team_perf = df.groupby('RaceTeam')['FinalPosition'].mean().reset_index().rename(columns={'FinalPosition':'TeamAvgPosition'})
    df = pd.merge(df, team_perf, on='RaceTeam', how='left')
    # One-hot encode EventName.
    df = pd.get_dummies(df, columns=['EventName'], prefix='Event')
    return df

if __name__=="__main__":
    df = pd.read_csv("f1_current_season_data.csv")
    df = engineer_features(df)
    print(df.head())
    df.to_csv("f1_current_season_features.csv", index=False)
