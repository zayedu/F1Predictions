#!/usr/bin/env python3

import datetime
import fastf1
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
import pytz

# Import the data collection functions from your existing data_collection.py
# (Ensure data_collection.py is in the same directory or PYTHONPATH.)
from data_collection import collect_season_data

def update_last_season_csv():
    """
    Collects and saves last season's (2023) data to f1_2023_data.csv.
    This function can be run manually or scheduled as needed.
    """
    eastern = pytz.timezone("US/Eastern")
    now = datetime.datetime.now(eastern)
    year = now.year - 1
    print(f"Collecting season data for {year} (last season)...")
    season_df = collect_season_data(year)
    if season_df.empty:
        print("No data collected for last season.")
    else:
        output_file = f"f1_{year}_data.csv"
        season_df.to_csv(output_file, index=False)
        print(f"Last season data saved to {output_file} with {len(season_df)} rows.")

def update_current_season_csv():
    """
    Collects current season data (for all completed rounds) and saves it to
    f1_current_season_data.csv.
    """
    # Use the Eastern timezone for consistency with the scheduled time
    eastern = pytz.timezone("US/Eastern")
    now = datetime.datetime.now(eastern)
    current_year = now.year

    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Updating current season data for {current_year}...")

    season_df = collect_season_data(current_year)
    if season_df.empty:
        print("No current season data available.")
    else:
        output_file = "f1_current_season_data.csv"
        season_df.to_csv(output_file, index=False)
        print(f"Current season data updated and saved to {output_file} with {len(season_df)} rows.")

def main():
    # Enable FastF1 caching (optional but recommended)
    fastf1.Cache.enable_cache('./fastf1_cache')

    # (Optional) Update last season's CSV once on startup.
    #update_last_season_csv()
    #update_current_season_csv()

    # Create a scheduler with the US/Eastern timezone
    scheduler = BlockingScheduler(timezone="US/Eastern")

    # Schedule the current season update job to run every Monday at 12pm EST.
    scheduler.add_job(update_current_season_csv, 'cron', day_of_week='mon', hour=15, minute=0)

    print("Scheduler started. The current season CSV will be updated every Monday at 12pm EST.")
    try:
        scheduler.start()
        print('Scheduler started')
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

if __name__ == "__main__":
    main()
