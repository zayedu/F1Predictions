#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime

def collect_current_season_data(year):
    """
    Collects current season data (year) for all races with EventDate < today.
    For each completed round, loads the Qualifying (Q) session and the Race session,
    computes the best qualifying lap per driver, merges with race results,
    and returns a DataFrame.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    today = datetime.datetime.now().date()
    completed = schedule[schedule['EventDate'] < today]
    all_data = []
    for idx, event in completed.iterrows():
        round_number = int(event['RoundNumber'])
        event_name = event['EventName']
        print(f"Collecting data for Round {round_number} - {event_name}")
        try:
            # Load Qualifying session
            session_q = fastf1.get_session(year, round_number, 'Q')
            session_q.load()
            laps_q = session_q.laps
            if laps_q.empty:
                print(f"No qualifying data for Round {round_number}")
                continue
            # Best lap per driver from Q
            best_lap = laps_q.groupby('Driver', as_index=False)['LapTime'].min().rename(columns={'LapTime': 'BestQualiLap'})
            best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})

            # Load Race session results
            session_r = fastf1.get_session(year, round_number, 'R')
            session_r.load()
            race_results = session_r.results
            if race_results is None or race_results.empty:
                print(f"No race data for Round {round_number}")
                continue
            race_results = race_results.rename(columns={'Position': 'FinalPosition', 'TeamName': 'RaceTeam'})
            keep = ['Abbreviation', 'FullName', 'RaceTeam', 'FinalPosition', 'GridPosition', 'Points', 'Status']
            race_results = race_results[keep]

            merged = pd.merge(best_lap, race_results, on='Abbreviation', how='inner')
            merged['Year'] = year
            merged['RoundNumber'] = round_number
            merged['EventName'] = event_name
            all_data.append(merged)
        except Exception as e:
            print(f"Error collecting data for Round {round_number}: {e}")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    year = 2025
    df = collect_current_season_data(year)
    print("Collected current season data:")
    print(df.head())
    df.to_csv("f1_current_season_data.csv", index=False)
