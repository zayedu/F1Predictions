#!/usr/bin/env python3
import pandas as pd
import numpy as np
from fastf1 import utils
import fastf1
import datetime

def convert_time_to_seconds(time_val):
    try:
        return utils.delta_to_seconds(time_val)
    except Exception:
        try:
            return pd.to_timedelta(time_val).total_seconds()
        except Exception:
            return np.nan

def create_best_quali_time(row):
    """
    Creates a unified qualifying lap time (in seconds).
    Prefer Q_BestQualiLap if available; otherwise, FP3_BestQualiLap.
    """
    q_time = row.get("Q_BestQualiLap")
    fp3_time = row.get("FP3_BestQualiLap")
    if pd.notnull(q_time):
        return convert_time_to_seconds(q_time)
    elif pd.notnull(fp3_time):
        return convert_time_to_seconds(fp3_time)
    else:
        return np.nan

def process_qualifying_session(year, round_number, event_name):
    session = fastf1.get_session(year, round_number, 'Q')
    session.load(laps=True, weather=True)
    laps = session.laps
    if laps.empty:
        return pd.DataFrame()
    best_lap = laps.groupby('Driver', as_index=False)['LapTime'].min().rename(
        columns={'LapTime': 'Q_BestQualiLap'}
    )
    best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            numeric_cols = weather.select_dtypes(include=['number']).columns
            weather_mean = weather[numeric_cols].mean()
        else:
            weather_mean = pd.Series(dtype=float)
    except Exception:
        weather_mean = pd.Series(dtype=float)
    for col in weather_mean.index:
        best_lap[f"Q_Weather_{col}"] = weather_mean[col]
    best_lap['Year'] = year
    best_lap['RoundNumber'] = round_number
    best_lap['EventName'] = event_name
    best_lap['SessionType'] = "Q"
    return best_lap

def process_fp3_session(year, round_number, event_name):
    session = fastf1.get_session(year, round_number, 'FP3')
    try:
        session.load(laps=True, weather=True)
    except Exception as e:
        print(f"FP3 load failed for Round {round_number} ({event_name}): {e}")
        return pd.DataFrame()
    laps = session.laps
    if laps.empty:
        return pd.DataFrame()
    best_lap = laps.groupby('Driver', as_index=False)['LapTime'].min().rename(
        columns={'LapTime': 'FP3_BestQualiLap'}
    )
    best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            numeric_cols = weather.select_dtypes(include=['number']).columns
            weather_mean = weather[numeric_cols].mean()
        else:
            weather_mean = pd.Series(dtype=float)
    except Exception:
        weather_mean = pd.Series(dtype=float)
    for col in weather_mean.index:
        best_lap[f"FP3_Weather_{col}"] = weather_mean[col]
    best_lap['Year'] = year
    best_lap['RoundNumber'] = round_number
    best_lap['EventName'] = event_name
    best_lap['SessionType'] = "FP3"
    return best_lap

def process_race_session(year, round_number, event_name):
    session = fastf1.get_session(year, round_number, 'R')
    try:
        session.load(laps=True)
    except Exception as e:
        print(f"Race session load failed for Round {round_number} ({event_name}): {e}")
        return pd.DataFrame()
    race_results = session.results
    if race_results is None or race_results.empty:
        return pd.DataFrame()
    race_results = race_results.rename(columns={'Position': 'R_FinalPosition', 'TeamName': 'RaceTeam'})
    keep = ['Abbreviation', 'FullName', 'RaceTeam', 'R_FinalPosition', 'GridPosition', 'Points', 'Status']
    race_results = race_results[keep]
    race_results['Year'] = year
    race_results['RoundNumber'] = round_number
    race_results['EventName'] = event_name
    race_results['SessionType'] = "R"
    return race_results

def merge_sessions(q_df, fp3_df, r_df):
    """Merges available Q, FP3, and Race session DataFrames on common keys."""
    merged = r_df.copy() if not r_df.empty else pd.DataFrame()
    if not q_df.empty:
        if merged.empty:
            merged = q_df.copy()
        else:
            merged = pd.merge(merged, q_df.drop(columns=['Year','RoundNumber','EventName','SessionType']),
                              on='Abbreviation', how='left', suffixes=('', '_Q'))
    if not fp3_df.empty:
        if merged.empty:
            merged = fp3_df.copy()
        else:
            merged = pd.merge(merged, fp3_df.drop(columns=['Year','RoundNumber','EventName','SessionType']),
                              on='Abbreviation', how='left', suffixes=('', '_FP3'))
    return merged

def get_last_season_data(last_year):
    """
    Retrieves last season’s combined session data live from the API.
    Loops over the event schedule for last_year, skips testing events,
    and for each round, attempts to process Q, FP3, and Race sessions.
    Rounds that fail (for example, due to missing FP3 sessions) are skipped.
    Returns a DataFrame of merged data.
    """
    schedule = fastf1.get_event_schedule(last_year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    all_events = []
    # In last season, all events are completed.
    for _, event in schedule.iterrows():
        round_number = int(event['RoundNumber'])
        event_name = event['EventName']
        if "Test" in event_name or "Testing" in event_name or "Pre-Season" in event_name:
            print(f"[Last Season] Skipping testing event: {event_name}")
            continue
        print(f"[Last Season] Processing Round {round_number} - {event_name}")
        try:
            q_df = process_qualifying_session(last_year, round_number, event_name)
        except Exception as e:
            print(f"[Last Season] Error processing Q session for Round {round_number}: {e}")
            q_df = pd.DataFrame()
        try:
            fp3_df = process_fp3_session(last_year, round_number, event_name)
        except Exception as e:
            print(f"[Last Season] Error processing FP3 session for Round {round_number}: {e}")
            fp3_df = pd.DataFrame()
        try:
            r_df = process_race_session(last_year, round_number, event_name)
        except Exception as e:
            print(f"[Last Season] Error processing Race session for Round {round_number}: {e}")
            r_df = pd.DataFrame()
        merged = merge_sessions(q_df, fp3_df, r_df)
        if not merged.empty:
            all_events.append(merged)
    if not all_events:
        print("[Last Season] No event data collected.")
        return pd.DataFrame()
    return pd.concat(all_events, ignore_index=True)

def engineer_features(current_csv, last_season_csv=None):
    """
    Performs extended feature engineering using the combined current season data from current_csv
    and last season data retrieved live if last_season_csv is not provided.
    Creates:
      - BestQualiLap_s: unified lap time (prefers Q_BestQualiLap then FP3_BestQualiLap)
      - CurrentDriverForm: average R_FinalPosition per driver (current season)
      - TeamAvgPosition: average R_FinalPosition per team (current season)
      - Weather_code: extracted numeric weather value (using Q_Weather_Temperature if available)
      - One-hot encoded EventName dummy columns.
      - If last season data is available, computes LastDriverForm, LastTeamPerf, and LastTrackPerf.
    """
    df = pd.read_csv(current_csv)
    df["Abbreviation"] = df["Abbreviation"].str.upper()

    # Create unified qualifying lap time.
    df["BestQualiLap_s"] = df.apply(create_best_quali_time, axis=1)

    # Compute current driver form.
    if "R_FinalPosition" in df.columns:
        current_form = df.groupby("Abbreviation")["R_FinalPosition"].mean().reset_index().rename(
            columns={"R_FinalPosition": "CurrentDriverForm"}
        )
        df = pd.merge(df, current_form, on="Abbreviation", how="left")
    else:
        df["CurrentDriverForm"] = np.nan

    # Compute current team performance.
    if "RaceTeam" in df.columns and "R_FinalPosition" in df.columns:
        team_perf = df.groupby("RaceTeam")["R_FinalPosition"].mean().reset_index().rename(
            columns={"R_FinalPosition": "TeamAvgPosition"}
        )
        df = pd.merge(df, team_perf, on="RaceTeam", how="left")
    else:
        df["TeamAvgPosition"] = np.nan

    # Extract weather feature: use Q_Weather_Temperature if available.
    if "Q_Weather_Temperature" in df.columns:
        df["Weather_code"] = df["Q_Weather_Temperature"].fillna(0).astype(float)
    else:
        df["Weather_code"] = 0

    # One-hot encode EventName.
    if "EventName" in df.columns:
        df = pd.get_dummies(df, columns=["EventName"], prefix="Event")

    # Ensure RoundNumber is numeric.
    df["RoundNumber"] = pd.to_numeric(df["RoundNumber"], errors='coerce')

    # --- Incorporate Last Season Data ---
    last_year = 2024
    ls_df = None
    if last_season_csv is not None:
        try:
            ls_df = pd.read_csv(last_season_csv)
        except Exception as e:
            print(f"Error loading last season CSV: {e}")
            ls_df = get_last_season_data(last_year)
    else:
        ls_df = get_last_season_data(last_year)

    if ls_df.empty:
        print("[Last Season] No data available.")
        df["LastDriverForm"] = np.nan
        df["LastTeamPerf"] = np.nan
        df["LastTrackPerf"] = np.nan
    else:
        if "R_FinalPosition" in ls_df.columns:
            ls_form = ls_df.groupby("Abbreviation")["R_FinalPosition"].mean().reset_index().rename(
                columns={"R_FinalPosition": "LastDriverForm"}
            )
        else:
            ls_form = pd.DataFrame(columns=["Abbreviation", "LastDriverForm"])
        if "RaceTeam" in ls_df.columns and "R_FinalPosition" in ls_df.columns:
            ls_team = ls_df.groupby("RaceTeam")["R_FinalPosition"].mean().reset_index().rename(
                columns={"R_FinalPosition": "LastTeamPerf"}
            )
        else:
            ls_team = pd.DataFrame(columns=["RaceTeam", "LastTeamPerf"])
        if "EventName" in ls_df.columns and "R_FinalPosition" in ls_df.columns:
            track_perf = ls_df.groupby(["Abbreviation", "EventName"])["R_FinalPosition"].mean().reset_index().rename(
                columns={"R_FinalPosition": "LastTrackPerf"}
            )
        else:
            track_perf = pd.DataFrame(columns=["Abbreviation", "LastTrackPerf"])
        df = pd.merge(df, ls_form, on="Abbreviation", how="left")
        if not ls_team.empty and "RaceTeam" in df.columns:
            df = pd.merge(df, ls_team, on="RaceTeam", how="left")
        if not track_perf.empty:
            df = pd.merge(df, track_perf, on="Abbreviation", how="left")

    keep_cols = ["Abbreviation", "Year", "RoundNumber", "BestQualiLap_s",
                 "CurrentDriverForm", "LastDriverForm", "TeamAvgPosition", "LastTeamPerf", "LastTrackPerf", "Weather_code"]
    event_cols = [col for col in df.columns if col.startswith("Event_")]
    keep_cols.extend(event_cols)
    final_df = df[keep_cols]
    return final_df

if __name__=="__main__":
    current_csv = "f1_current_season_combined.csv"
    # No last season CSV provided; data will be retrieved live.
    features_df = engineer_features(current_csv, last_season_csv=None)
    print("Engineered extended features (first 5 rows):")
    print(features_df.head())
    features_df.to_csv("f1_current_season_features_extended.csv", index=False)
