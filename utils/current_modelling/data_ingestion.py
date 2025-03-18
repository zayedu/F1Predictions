#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime

def collect_current_season_data(year):
    """
    Collects current season data for all races that have completed (EventDate < today).
    For each race, loads the Qualifying (Q) and Race sessions, extracts the best qualifying lap,
    retrieves the weather from the Qualifying session, and merges with race results.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    today = datetime.datetime.now().date()
    completed = schedule[schedule['EventDate'] < today]

    all_data = []
    for _, event in completed.iterrows():
        round_number = int(event['RoundNumber'])
        event_name = event['EventName']
        print(f"Collecting data for Round {round_number} - {event_name}")
        try:
            # Load Qualifying session.
            session_q = fastf1.get_session(year, round_number, 'Q')
            session_q.load()
            laps_q = session_q.laps
            if laps_q.empty:
                print(f"No qualifying data for Round {round_number}; skipping.")
                continue
            # Compute best qualifying lap per driver.
            best_lap = laps_q.groupby('Driver', as_index=False)['LapTime'].min().rename(
                columns={'LapTime': 'BestQualiLap'}
            )
            best_lap = best_lap.rename(columns={'Driver': 'Abbreviation'})

            # Retrieve weather from the Qualifying session.
            try:
                weather = session_q.weather
            except Exception:
                weather = "Clear"  # default weather

            # Load Race session data.
            session_r = fastf1.get_session(year, round_number, 'R')
            session_r.load()
            race_results = session_r.results
            if race_results is None or race_results.empty:
                print(f"No race data for Round {round_number}; skipping.")
                continue
            race_results = race_results.rename(columns={'Position': 'FinalPosition', 'TeamName': 'RaceTeam'})
            keep_cols = ['Abbreviation', 'FullName', 'RaceTeam', 'FinalPosition', 'GridPosition', 'Points', 'Status']
            race_results = race_results[keep_cols]

            # Merge qualifying best lap and race results.
            merged = pd.merge(best_lap, race_results, on='Abbreviation', how='inner')
            merged['Year'] = year
            merged['RoundNumber'] = round_number
            merged['EventName'] = event_name
            merged['Weather'] = weather
            all_data.append(merged)
        except Exception as e:
            print(f"Error collecting data for Round {round_number}: {e}")

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    YEAR = datetime.datetime.now().year
    df = collect_current_season_data(YEAR)
    print("Collected current season data (first 5 rows):")
    print(df.head())
    df.to_csv("f1_current_season_data.csv", index=False)
    print("Data saved to f1_current_season_data.csv")
