#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime

def process_qualifying_session(year, round_number, event_name):
    """Load Q session, compute best lap and average weather info; prefix columns with 'Q_'."""
    session = fastf1.get_session(year, round_number, 'Q')
    session.load(laps=True, weather=True)
    laps = session.laps
    if laps.empty:
        print(f"No qualifying lap data for Round {round_number}")
        return pd.DataFrame()

    # Best lap per driver.
    best_lap = laps.groupby('Driver', as_index=False)['LapTime'].min().rename(
        columns={'LapTime': 'BestQualiLap'}
    )
    best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})
    # Prefix Q columns.
    best_lap = best_lap.rename(columns={'BestQualiLap': 'Q_BestQualiLap'})

    # Retrieve weather data.
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            numeric_cols = weather.select_dtypes(include=['number']).columns
            weather_mean = weather[numeric_cols].mean()
        else:
            print(f"No weather data in Q session for Round {round_number}.")
            weather_mean = pd.Series(dtype=float)
    except Exception as e:
        print(f"Error retrieving weather from Q session for Round {round_number}: {e}")
        weather_mean = pd.Series(dtype=float)

    # Create a DataFrame with weather info.
    for col in weather_mean.index:
        best_lap[f"Q_Weather_{col}"] = weather_mean[col]

    # Add keys.
    best_lap['Year'] = year
    best_lap['RoundNumber'] = round_number
    best_lap['EventName'] = event_name
    return best_lap

def process_fp3_session(year, round_number, event_name):
    """Load FP3 session; compute best lap and weather info; prefix columns with 'FP3_'."""
    session = fastf1.get_session(year, round_number, 'FP3')
    session.load(laps=True, weather=True)
    laps = session.laps
    if laps.empty:
        print(f"No FP3 lap data for Round {round_number}")
        return pd.DataFrame()

    best_lap = laps.groupby('Driver', as_index=False)['LapTime'].min().rename(
        columns={'LapTime': 'BestQualiLap'}
    )
    best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})
    best_lap = best_lap.rename(columns={'BestQualiLap': 'FP3_BestQualiLap'})

    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            numeric_cols = weather.select_dtypes(include=['number']).columns
            weather_mean = weather[numeric_cols].mean()
        else:
            print(f"No weather data in FP3 session for Round {round_number}.")
            weather_mean = pd.Series(dtype=float)
    except Exception as e:
        print(f"Error retrieving weather from FP3 session for Round {round_number}: {e}")
        weather_mean = pd.Series(dtype=float)

    for col in weather_mean.index:
        best_lap[f"FP3_Weather_{col}"] = weather_mean[col]

    best_lap['Year'] = year
    best_lap['RoundNumber'] = round_number
    best_lap['EventName'] = event_name
    return best_lap

def process_race_session(year, round_number, event_name):
    """Load Race session data; rename key columns with prefix 'R_'."""
    session = fastf1.get_session(year, round_number, 'R')
    session.load(laps=True)
    race_results = session.results
    if race_results is None or race_results.empty:
        print(f"No race data for Round {round_number}")
        return pd.DataFrame()

    race_results = race_results.rename(columns={'Position': 'FinalPosition', 'TeamName': 'RaceTeam'})
    keep = ['Abbreviation', 'FullName', 'RaceTeam', 'FinalPosition', 'GridPosition', 'Points', 'Status']
    race_results = race_results[keep]
    # Prefix race columns.
    race_results = race_results.rename(columns={
        'FinalPosition': 'R_FinalPosition',
        'GridPosition': 'R_GridPosition',
        'Points': 'R_Points',
        'Status': 'R_Status'
    })
    race_results['Year'] = year
    race_results['RoundNumber'] = round_number
    race_results['EventName'] = event_name
    race_results['SessionType'] = "R"
    return race_results

def merge_sessions(q_df, fp3_df, r_df):
    """
    Merges Q, FP3, and Race session DataFrames on common keys.
    Returns one row per driver for the Grand Prix.
    """
    # Start with Race data (usually the most complete for identity info).
    merged = r_df.copy()
    # Merge with Q data.
    if not q_df.empty:
        merged = pd.merge(merged, q_df.drop(columns=['Year','RoundNumber','EventName']),
                          on='Abbreviation', how='left', suffixes=('', '_Q'))
    # Merge with FP3 data.
    if not fp3_df.empty:
        merged = pd.merge(merged, fp3_df.drop(columns=['Year','RoundNumber','EventName']),
                          on='Abbreviation', how='left', suffixes=('', '_FP3'))
    return merged

def collect_all_sessions(year):
    """
    Processes all completed races for the current season.
    For each event (skipping testing events), processes Q, FP3, and Race sessions,
    then merges them into one row per driver per event.
    Returns a combined DataFrame.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    today = datetime.datetime.now().date()
    completed = schedule[schedule['EventDate'] < today]

    all_events = []
    for _, event in completed.iterrows():
        round_number = int(event['RoundNumber'])
        event_name = event['EventName']
        # Skip testing events.
        if "Test" in event_name or "Testing" in event_name:
            print(f"Skipping testing event: {event_name}")
            continue
        print(f"Processing Round {round_number} - {event_name}")

        q_df = process_qualifying_session(year, round_number, event_name)
        fp3_df = process_fp3_session(year, round_number, event_name)
        r_df = process_race_session(year, round_number, event_name)

        # Merge sessions (outer merge so that if one session is missing, data from others remain).
        merged = merge_sessions(q_df, fp3_df, r_df)
        all_events.append(merged)

    if not all_events:
        print("No event data collected for the current season.")
        return pd.DataFrame()
    return pd.concat(all_events, ignore_index=True)

if __name__ == "__main__":
    YEAR = 2025
    combined_data = collect_all_sessions(YEAR)
    print("Combined data (first 5 rows):")
    print(combined_data.head())
    combined_data.to_csv("f1_current_season_combined.csv", index=False)
    print("Data saved to f1_current_season_combined.csv")
