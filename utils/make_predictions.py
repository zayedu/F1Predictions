#!/usr/bin/env python3
import fastf1
import pandas as pd
import datetime
import joblib
import numpy as np

############################################
# Helper Functions
############################################

def get_next_race_info(year=2025):
    """
    Retrieves the FastF1 event schedule for the given year (2025),
    converts EventDate to a date, and returns the next upcoming race.
    """
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=['RoundNumber'])
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.date
    today = datetime.datetime.now().date()
    upcoming = schedule[schedule['EventDate'] >= today]
    if upcoming.empty:
        print("No upcoming races found for the year.")
        return None
    upcoming = upcoming.sort_values('EventDate')
    next_race = upcoming.iloc[0]
    return next_race

def get_current_driver_list(year=2025, round_number=None):
    """
    Retrieves the current driver list from the FP1 session for the given season.
    If the 'drivers' attribute is a dict, returns its keys as a list;
    if it is already a list, returns it as-is.
    """
    try:
        fp1_session = fastf1.get_session(year, round_number, 'FP1')
        fp1_session.load()
        drivers = fp1_session.drivers
        if isinstance(drivers, dict):
            return list(drivers.keys())
        elif isinstance(drivers, list):
            return drivers
        else:
            return []
    except Exception as e:
        print("Error loading current driver list from FP session:", e)
        return []

def adjust_team_performance(df, current_df):
    """
    Adjusts team performance in a driver DataFrame by merging in current season data.
    For each driver, we:
      1. Get the driver's current team from current season data.
      2. Get the current team's average finishing position (TeamAvgPosition).
      3. Replace the team performance feature with the current team's value.
    """
    try:
        current_df["RaceTeam"] = current_df["RaceTeam"].str.upper().str.strip()
        current_driver_team = current_df.groupby("Abbreviation")["RaceTeam"] \
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]) \
            .reset_index().rename(columns={"RaceTeam": "CurrentTeam"})
        df["Abbreviation"] = df["Abbreviation"].str.upper().str.strip()
        df = pd.merge(df, current_driver_team, on="Abbreviation", how="left")
        current_team_perf = current_df.groupby("RaceTeam")["FinalPosition"] \
            .mean().reset_index().rename(columns={"FinalPosition": "TeamAvgPosition_current"})
        df = pd.merge(df, current_team_perf, left_on="CurrentTeam", right_on="RaceTeam", how="left")
        df["TeamAvgPosition"] = df["TeamAvgPosition_current"].fillna(df.get("TeamAvgPosition", np.nan))
        df.drop(columns=["CurrentTeam", "RaceTeam", "TeamAvgPosition_current"], inplace=True, errors="ignore")
    except Exception as e:
        print("Error in adjusting team performance:", e)
    return df

def add_missing_current_drivers(grouped, current_driver_list, current_df, hist_event, next_race, event_col):
    """
    Ensures that all current drivers (from API) appear in the DataFrame.
    For any driver in the current grid that is missing from grouped historical data,
    creates a new row using overall historical averages (or overall defaults) and current season data.
    """
    missing = set(current_driver_list) - set(grouped["Abbreviation"].unique())
    if not missing:
        return grouped
    # Use overall historical averages if available; if hist_event is empty, use overall averages from hist_df.
    if not hist_event.empty:
        overall_avg_s1 = hist_event["AvgSector1"].apply(lambda x: float(x) if pd.notnull(x) else np.nan).mean()
        overall_avg_s2 = hist_event["AvgSector2"].apply(lambda x: float(x) if pd.notnull(x) else np.nan).mean()
        overall_avg_s3 = hist_event["AvgSector3"].apply(lambda x: float(x) if pd.notnull(x) else np.nan).mean()
    else:
        overall_avg_s1 = overall_avg_s2 = overall_avg_s3 = np.nan

    try:
        driver_form = current_df.groupby(['Year', 'Abbreviation'])['FinalPosition'] \
            .mean().reset_index().rename(columns={'FinalPosition': 'DriverForm'})
        team_perf = current_df.groupby("RaceTeam")["FinalPosition"].mean().reset_index() \
            .rename(columns={"FinalPosition": "TeamAvgPosition"})
        current_df["RaceTeam"] = current_df["RaceTeam"].str.upper().str.strip()
        current_driver_team = current_df.groupby("Abbreviation")["RaceTeam"] \
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]) \
            .reset_index().rename(columns={"RaceTeam": "CurrentTeam"})
    except Exception as e:
        print("Error processing current season driver info for missing drivers:", e)
        return grouped

    missing_rows = []
    for drv in missing:
        dform_row = driver_form[driver_form["Abbreviation"] == drv]
        dform_val = dform_row["DriverForm"].values[0] if not dform_row.empty else np.nan
        curr_team_series = current_driver_team[current_driver_team["Abbreviation"] == drv]["CurrentTeam"]
        curr_team = curr_team_series.values[0] if not curr_team_series.empty else ""
        team_row = team_perf[team_perf["RaceTeam"] == curr_team]
        team_val = team_row["TeamAvgPosition"].values[0] if not team_row.empty else np.nan
        missing_rows.append({
            "Abbreviation": drv,
            "AvgSector1": overall_avg_s1,
            "AvgSector2": overall_avg_s2,
            "AvgSector3": overall_avg_s3,
            "DriverForm": dform_val,
            "TeamAvgPosition": team_val,
            "RoundNumber": next_race["RoundNumber"],
            "Weather_code": 0,
            event_col: 1
        })
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        for col in ["AvgSector1", "AvgSector2", "AvgSector3"]:
            missing_df[col + "_s"] = missing_df[col]
        grouped = pd.concat([grouped, missing_df], ignore_index=True)
    return grouped

def predict_qualifying_from_fp3(next_race, qualifying_model):
    """
    Uses FP3 session data for the next race to compute features:
      - Best FP3 lap and average sector times.
      - Merges in current-season performance (driver form and team performance).
      - Adjusts team performance based on current team assignments.
      - Filters results to only include drivers on the current grid.
      - Returns a DataFrame with predicted qualifying lap times and ranking.
    """
    try:
        fp3_session = fastf1.get_session(2025, next_race["RoundNumber"], 'FP3')
        fp3_session.load()
        fp3_laps = fp3_session.laps
        if fp3_laps.empty:
            print("FP3 session is empty.")
            return None
        best_laps_fp3 = fp3_laps.groupby('Driver', as_index=False)['LapTime'] \
            .min().rename(columns={'LapTime': 'BestFP3Lap'})
        avg_sectors_fp3 = fp3_laps.groupby('Driver', as_index=False).agg({
            'Sector1Time': 'mean',
            'Sector2Time': 'mean',
            'Sector3Time': 'mean',
            'Team': 'first'
        }).rename(columns={
            'Sector1Time': 'AvgSector1',
            'Sector2Time': 'AvgSector2',
            'Sector3Time': 'AvgSector3',
            'Team': 'QualiTeam'
        })
        fp3_data = pd.merge(best_laps_fp3, avg_sectors_fp3, on='Driver', how='left')
        fp3_data = fp3_data.rename(columns={'Driver': 'Abbreviation'})
    except Exception as e:
        print("Error loading FP3 session for upcoming race:", e)
        return None

    try:
        current_df = pd.read_csv("f1_current_season_data.csv")
        current_df["RaceTeam"] = current_df["RaceTeam"].str.upper().str.strip()
        driver_form = current_df.groupby(['Year', 'Abbreviation'])['FinalPosition'] \
            .mean().reset_index().rename(columns={'FinalPosition': 'DriverForm'})
        fp3_data = pd.merge(fp3_data, driver_form, on='Abbreviation', how='left')
        fp3_data["QualiTeam"] = fp3_data["QualiTeam"].str.upper().str.strip()
    except Exception as e:
        print("Error loading current season performance data:", e)
        return None

    fp3_data = adjust_team_performance(fp3_data, current_df)

    for col in ['AvgSector1', 'AvgSector2', 'AvgSector3']:
        new_col = col + "_s"
        fp3_data[new_col] = fp3_data[col].apply(lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan)

    try:
        weather = fp3_session.weather
        weather_mapping = {"Clear": 0, "Sunny": 0, "Cloudy": 1, "Overcast": 1, "Rain": 2, "Rainy": 2, "Wet": 3}
        fp3_data['Weather_code'] = weather_mapping.get(weather, 0)
    except Exception:
        fp3_data['Weather_code'] = 0

    fp3_data['RoundNumber'] = next_race["RoundNumber"]
    event_col = "Event_" + str(next_race["EventName"]).replace(" ", "_")
    fp3_data[event_col] = 1

    base_features = ["AvgSector1_s", "AvgSector2_s", "AvgSector3_s",
                     "DriverForm", "TeamAvgPosition", "Weather_code", "RoundNumber"]
    features = base_features + [event_col]
    X_new = fp3_data[features]
    expected_features = qualifying_model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    fp3_data["PredictedQualiLap_s"] = qualifying_model.predict(X_new)
    fp3_data = fp3_data.sort_values("PredictedQualiLap_s")
    fp3_data["PredictedQualiRank"] = range(1, len(fp3_data) + 1)

    # Filter to include only current drivers.
    current_driver_list = get_current_driver_list(2025, next_race["RoundNumber"])
    fp3_data = fp3_data[fp3_data["Abbreviation"].isin(current_driver_list)]
    return fp3_data

def predict_qualifying_from_historical(next_race, qualifying_model):
    """
    Falls back to using historical data (2024) for the same event.
    Builds a feature set from last year's data, adjusts team performance based on current season data,
    and then ensures that all current drivers (from the API) are included.
    If no event-specific historical data exists (e.g., for a new event), overall historical averages are used.
    """
    try:
        hist_df = pd.read_csv("f1_2024_data.csv")
        if "BestQualiLap_s" not in hist_df.columns:
            hist_df["BestQualiLap_s"] = hist_df["BestQualiLap"].apply(
                lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan
            )
    except Exception as e:
        print("Error loading historical data:", e)
        return None

    event_name = next_race["EventName"]
    hist_event = hist_df[hist_df["EventName"] == event_name]
    event_col = "Event_" + str(event_name).replace(" ", "_")
    if hist_event.empty:
        print(f"No event-specific historical data available for {event_name}. Using overall historical averages.")
        overall_avg_s1 = hist_df["AvgSector1"].apply(lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan).mean()
        overall_avg_s2 = hist_df["AvgSector2"].apply(lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan).mean()
        overall_avg_s3 = hist_df["AvgSector3"].apply(lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan).mean()
        overall_driver_form = hist_df["FinalPosition"].mean()
        new_rows = []
        current_driver_list = get_current_driver_list(2025, next_race["RoundNumber"])
        for drv in current_driver_list:
            new_rows.append({
                "Abbreviation": drv,
                "AvgSector1": overall_avg_s1,
                "AvgSector2": overall_avg_s2,
                "AvgSector3": overall_avg_s3,
                "DriverForm": overall_driver_form,
                "TeamAvgPosition": overall_driver_form,  # fallback
                "RoundNumber": next_race["RoundNumber"],
                "Weather_code": 0,
                event_col: 1
            })
        grouped = pd.DataFrame(new_rows)
        for col in ["AvgSector1", "AvgSector2", "AvgSector3"]:
            grouped[col + "_s"] = grouped[col]
    else:
        for col in ["AvgSector1", "AvgSector2", "AvgSector3"]:
            if hist_event[col].dtype == object:
                hist_event = hist_event.copy()
                try:
                    hist_event[col] = pd.to_timedelta(hist_event[col]).dt.total_seconds()
                except Exception as e:
                    print(f"Error converting {col}: {e}")
                    hist_event[col] = np.nan

        grouped = hist_event.groupby("Abbreviation").agg({
            "AvgSector1": "mean",
            "AvgSector2": "mean",
            "AvgSector3": "mean",
            "FinalPosition": "mean"
        }).reset_index().rename(columns={"FinalPosition": "DriverForm"})
        grouped["TeamAvgPosition"] = hist_event["FinalPosition"].mean()
        for col in ["AvgSector1", "AvgSector2", "AvgSector3"]:
            grouped[col + "_s"] = grouped[col]
        grouped["RoundNumber"] = next_race["RoundNumber"]
        grouped["Weather_code"] = 0
        grouped[event_col] = 1

    # Merge current drivers missing from the historical group.
    current_driver_list = get_current_driver_list(2025, next_race["RoundNumber"])
    try:
        current_df = pd.read_csv("f1_current_season_data.csv")
        current_df["Abbreviation"] = current_df["Abbreviation"].str.upper().str.strip()
    except Exception as e:
        print("Error loading current season data for merging driver list:", e)
        current_df = pd.DataFrame()

    grouped = add_missing_current_drivers(grouped, current_driver_list, current_df, hist_event, next_race, event_col)

    try:
        grouped = adjust_team_performance(grouped, current_df)
    except Exception as e:
        print("Error adjusting team performance in historical fallback:", e)

    base_features = ["AvgSector1_s", "AvgSector2_s", "AvgSector3_s",
                     "DriverForm", "TeamAvgPosition", "Weather_code", "RoundNumber"]
    features = base_features + [event_col]
    X_new = grouped[features]
    expected_features = qualifying_model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    grouped["PredictedQualiLap_s"] = qualifying_model.predict(X_new)
    grouped = grouped.sort_values("PredictedQualiLap_s")
    grouped["PredictedQualiRank"] = range(1, len(grouped) + 1)
    grouped = grouped[grouped["Abbreviation"].isin(current_driver_list)]
    return grouped

def predict_race_standings(next_race, race_model, historical_data):
    """
    When qualifying has occurred:
      - Loads actual qualifying session data for the next race.
      - Converts qualifying lap times to seconds.
      - Merges with historical performance data (from 2024) for the same event.
      - Merges current-season performance from f1_current_season_data.csv.
      - Adjusts team performance based on current team assignments.
      - Constructs a feature set and uses the race model to predict race finishing positions.
      - Ranks drivers accordingly.
    """
    try:
        q_session = fastf1.get_session(2025, next_race["RoundNumber"], "Q")
        q_session.load()
        q_results = q_session.results
    except Exception as e:
        print("Error loading qualifying session for next race:", e)
        return None

    if q_results is None or q_results.empty:
        print("Qualifying session exists but no lap data is available.")
        return None

    q_results["BestQualiLap_s"] = q_results["BestQualiLap"].apply(
        lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan
    )

    event_name = next_race["EventName"]
    hist_event_data = historical_data[historical_data["EventName"] == event_name]
    if not hist_event_data.empty:
        hist_stats = hist_event_data.groupby("Abbreviation").agg({
            "BestQualiLap_s": "mean",
            "FinalPosition": "mean"
        }).rename(columns={"BestQualiLap_s": "PrevQualiLap", "FinalPosition": "PrevRacePos"}).reset_index()
    else:
        overall_prev_quali = 90
        overall_prev_race = 10
        hist_stats = pd.DataFrame(columns=["Abbreviation", "PrevQualiLap", "PrevRacePos"])

    race_features = pd.merge(q_results, hist_stats, on="Abbreviation", how="left")
    if not hist_event_data.empty:
        overall_prev_quali = hist_event_data["BestQualiLap_s"].mean()
        overall_prev_race = hist_event_data["FinalPosition"].mean()
    else:
        overall_prev_quali = 90
        overall_prev_race = 10
    race_features["PrevQualiLap"].fillna(overall_prev_quali, inplace=True)
    race_features["PrevRacePos"].fillna(overall_prev_race, inplace=True)

    try:
        current_df = pd.read_csv("f1_current_season_data.csv")
        current_df["RaceTeam"] = current_df["RaceTeam"].str.upper().str.strip()
        driver_form = current_df.groupby(['Year', 'Abbreviation'])['FinalPosition'] \
            .mean().reset_index().rename(columns={'FinalPosition': 'DriverForm'})
        race_features = pd.merge(race_features, driver_form, on='Abbreviation', how='left')
    except Exception as e:
        print("Error merging driver form from current season:", e)
        race_features["DriverForm"] = 6.0

    try:
        current_df = pd.read_csv("f1_current_season_data.csv")
        race_features = adjust_team_performance(race_features, current_df)
    except Exception as e:
        print("Error merging current season data for race prediction:", e)
        race_features["TeamAvgPosition"] = 5.0

    try:
        weather = q_session.weather
        weather_mapping = {"Clear": 0, "Sunny": 0, "Cloudy": 1, "Overcast": 1, "Rain": 2, "Rainy": 2, "Wet": 3}
        race_features["Weather_code"] = weather_mapping.get(weather, 0)
    except Exception:
        race_features["Weather_code"] = 0

    race_features["RoundNumber"] = next_race["RoundNumber"]
    event_col = "Event_" + str(event_name).replace(" ", "_")
    race_features[event_col] = 1

    base_features = ["BestQualiLap_s", "DriverForm", "TeamAvgPosition", "Weather_code", "RoundNumber", "PrevQualiLap", "PrevRacePos"]
    features = base_features + [event_col]
    X_new = race_features[features]
    expected_features = race_model.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    race_features["PredictedRacePos"] = race_model.predict(X_new)
    race_features = race_features.sort_values("PredictedRacePos")
    race_features["PredictedRaceRank"] = range(1, len(race_features) + 1)

    current_driver_list = get_current_driver_list(2025, next_race["RoundNumber"])
    race_features = race_features[race_features["Abbreviation"].isin(current_driver_list)]
    return race_features

############################################
# Main Prediction Logic
############################################

def main():
    current_year = 2025
    next_race = get_next_race_info(current_year)
    if next_race is None:
        print("No upcoming race found.")
        return
    print(f"Next race: Round {next_race['RoundNumber']} - {next_race['EventName']} on {next_race['EventDate']}")

    try:
        qualifying_model = joblib.load("qualifying_model.pkl")
        print("Qualifying model loaded.")
    except Exception as e:
        print("Error loading qualifying model:", e)
        return

    try:
        race_model = joblib.load("race_model.pkl")
        print("Race model loaded.")
    except Exception as e:
        print("Error loading race model:", e)
        race_model = None

    try:
        historical_data = pd.read_csv("f1_2024_data.csv")
        if "BestQualiLap_s" not in historical_data.columns:
            historical_data["BestQualiLap_s"] = historical_data["BestQualiLap"].apply(
                lambda x: pd.to_timedelta(x).total_seconds() if pd.notnull(x) else np.nan
            )
    except Exception as e:
        print("Error loading historical data:", e)
        historical_data = pd.DataFrame()

    try:
        q_session = fastf1.get_session(current_year, next_race["RoundNumber"], "Q")
        q_session.load()
        q_laps = q_session.laps
        qualifying_occurred = not q_laps.empty
    except Exception as e:
        print("Error loading qualifying session:", e)
        qualifying_occurred = False

    if not qualifying_occurred:
        print("Qualifying has not yet occurred for the next race.")
        quali_predictions = predict_qualifying_from_fp3(next_race, qualifying_model)
        if quali_predictions is None:
            print("FP3 data unavailable; falling back to historical data.")
            quali_predictions = predict_qualifying_from_historical(next_race, qualifying_model)
        if quali_predictions is not None:
            print("\nPredicted Qualifying Ranking:")
            print(quali_predictions[["Abbreviation", "PredictedQualiLap_s", "PredictedQualiRank"]])
        else:
            print("Unable to predict qualifying ranking using both FP3 and historical data.")
    else:
        print("Qualifying has occurred for the next race. Predicting race standings.")
        if race_model is None:
            print("Race model unavailable; cannot predict race standings.")
        else:
            race_predictions = predict_race_standings(next_race, race_model, historical_data)
            if race_predictions is not None:
                print("\nPredicted Race Standings:")
                print(race_predictions[["Abbreviation", "PredictedRacePos", "PredictedRaceRank"]])
            else:
                print("Unable to predict race standings.")

if __name__ == "__main__":
    main()
