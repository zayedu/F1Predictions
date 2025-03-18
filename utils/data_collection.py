#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime

def collect_qualifying_data(year: int, round_number: int) -> pd.DataFrame:
    """
    Fetches Qualifying data for a specific year and round_number.
    Returns a DataFrame with each driver's best lap time, average sector times,
    and an 'Abbreviation' column for merging with Race data.
    """
    # Load the Qualifying session
    session_q = fastf1.get_session(year, round_number, 'Q')
    session_q.load()
    laps_q = session_q.laps

    # Compute each driver's best Quali lap time
    best_laps_df = (
        laps_q.groupby('Driver', as_index=False)['LapTime']
        .min()
        .rename(columns={'LapTime': 'BestQualiLap'})
    )

    # Compute average sector times and retrieve the team name (first occurrence)
    avg_sectors_df = (
        laps_q.groupby('Driver', as_index=False)
        .agg({
            'Sector1Time': 'mean',
            'Sector2Time': 'mean',
            'Sector3Time': 'mean',
            'Team': 'first'
        })
        .rename(columns={
            'Sector1Time': 'AvgSector1',
            'Sector2Time': 'AvgSector2',
            'Sector3Time': 'AvgSector3',
            'Team': 'QualiTeam'
        })
    )

    # Merge best lap times with average sector times
    q_data = pd.merge(best_laps_df, avg_sectors_df, on='Driver', how='left')
    # Rename 'Driver' to 'Abbreviation' to match Race data
    q_data = q_data.rename(columns={'Driver': 'Abbreviation'})
    return q_data

def collect_race_data(year: int, round_number: int) -> pd.DataFrame:
    """
    Fetches Race data (final classification) for a specific year and round_number.
    Returns a DataFrame with each driver's final position, grid position, points, status, etc.
    """
    session_r = fastf1.get_session(year, round_number, 'R')
    session_r.load()
    race_results = session_r.results

    if race_results is None or race_results.empty:
        return pd.DataFrame()

    # Rename columns for consistency
    race_results = race_results.rename(columns={
        'Position': 'FinalPosition',
        'TeamName': 'RaceTeam'
    })

    # Keep only essential columns for merging and further analysis
    keep_cols = [
        'Abbreviation',  # e.g., 'HAM', 'VER', 'LEC'
        'FullName',
        'RaceTeam',
        'FinalPosition',
        'GridPosition',
        'Points',
        'Status'
    ]
    race_results = race_results[keep_cols]
    return race_results

def collect_season_data(year: int) -> pd.DataFrame:
    """
    Iterates over all rounds in the given year, collecting Qualifying and Race data.
    When the year is the current year, only events with an EventDate in the past are processed.
    Merges data by driver (using 'Abbreviation') and returns a combined DataFrame with metadata.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])

    # If we're collecting for the current season, filter out events that haven't occurred yet.
    if year == datetime.datetime.now().year:
        today = datetime.datetime.now().date()
        # Convert EventDate to a date object (the series is already datetime64[ns])
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
        schedule = schedule[schedule['EventDate'] < today]

    all_rounds_data = []

    for _, event in schedule.iterrows():
        round_number = int(event['RoundNumber'])
        event_name = event['EventName']
        print(f"Collecting data for Round {round_number} - {event_name}...")
        try:
            # Get Qualifying data
            q_data = collect_qualifying_data(year, round_number)
            if q_data.empty:
                print(f"  No Qualifying data found for round {round_number}. Skipping.")
                continue

            # Get Race data
            r_data = collect_race_data(year, round_number)
            if r_data.empty:
                print(f"  No Race data found for round {round_number}. Skipping.")
                continue

            # Merge data on 'Abbreviation'
            merged = pd.merge(q_data, r_data, on='Abbreviation', how='inner')
            merged['Year'] = year
            merged['RoundNumber'] = round_number
            merged['EventName'] = event_name

            # Reorder columns for clarity
            cols_order = [
                'Year', 'RoundNumber', 'EventName', 'Abbreviation', 'FullName',
                'QualiTeam', 'BestQualiLap', 'AvgSector1', 'AvgSector2', 'AvgSector3',
                'RaceTeam', 'FinalPosition', 'GridPosition', 'Points', 'Status'
            ]
            merged = merged[cols_order]
            all_rounds_data.append(merged)
            print(f"  -> {len(merged)} drivers merged for this round.")
        except Exception as e:
            print(f"  Error loading round {round_number}: {e}")
            continue

    if not all_rounds_data:
        print("No data collected for the season. Check if the schedule or sessions are valid.")
        return pd.DataFrame()

    final_df = pd.concat(all_rounds_data, ignore_index=True)
    return final_df

def main():
    # Enable caching to speed up repeated queries
    fastf1.Cache.enable_cache('./fastf1_cache')

    current_year = datetime.datetime.now().year
    print(f"Collecting season data for {current_year}...")
    season_df = collect_season_data(current_year)

    print("\nPreview of final merged DataFrame:")
    print(season_df.head())
    print(f"\nTotal rows collected: {len(season_df)}")

    # Save the collected data to a CSV file
    output_file = "f1_current_season_data.csv"
    season_df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

if __name__ == "__main__":
    main()
