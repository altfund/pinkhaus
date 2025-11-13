#!/usr/bin/env python3
"""
Dashboard Filter Control - Easy interface to set dashboard filters
"""

import os
import subprocess
import sys

FILTER_PRESETS = {
    "1": {
        "name": "English Soccer (Default)",
        "sports": "Soccer",
        "nations": "England,International,Europe",
        "leagues": ""
    },
    "2": {
        "name": "Top 5 European Leagues",
        "sports": "Soccer", 
        "nations": "England,Spain,Italy,Germany,France",
        "leagues": ""
    },
    "3": {
        "name": "All Soccer",
        "sports": "Soccer",
        "nations": "",
        "leagues": ""
    },
    "4": {
        "name": "US Sports",
        "sports": "Basketball,Baseball,American Football,Hockey",
        "nations": "USA",
        "leagues": ""
    },
    "5": {
        "name": "Premier League Only",
        "sports": "Soccer",
        "nations": "England",
        "leagues": "Premier League"
    },
    "6": {
        "name": "Champions League & International",
        "sports": "Soccer",
        "nations": "Europe,International",
        "leagues": "UEFA Champions League,Europa League,World Cup"
    },
    "7": {
        "name": "All Sports - No Filter",
        "sports": "",
        "nations": "",
        "leagues": ""
    }
}

def show_menu():
    print("\n=== Ominari Dashboard Filter Control ===")
    print("\nAvailable Filter Presets:")
    for key, preset in FILTER_PRESETS.items():
        print(f"  {key}. {preset['name']}")
    print("\n  c. Custom filters")
    print("  s. Show current filters")
    print("  q. Quit")
    
def get_current_filters():
    """Show current environment filter settings"""
    print("\n=== Current Filter Settings ===")
    print(f"ALLOWED_SPORTS: {os.environ.get('ALLOWED_SPORTS', '(not set - all sports)')}")
    print(f"ALLOWED_NATIONS: {os.environ.get('ALLOWED_NATIONS', '(not set - all nations)')}")
    print(f"ALLOWED_LEAGUES: {os.environ.get('ALLOWED_LEAGUES', '(not set - all leagues)')}")

def apply_filters(sports, nations, leagues):
    """Set environment variables and restart dashboard"""
    print("\n=== Applying Filters ===")
    
    # Set environment variables
    os.environ['ALLOWED_SPORTS'] = sports
    os.environ['ALLOWED_NATIONS'] = nations
    os.environ['ALLOWED_LEAGUES'] = leagues
    
    # Set database environment
    os.environ['PG_HOST'] = 'localhost'
    os.environ['PG_PORT'] = '5999'
    os.environ['PG_USER'] = 'ominari_user'
    os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
    os.environ['PG_DB'] = 'ominari_production'
    os.environ['USE_POSTGRESQL'] = '1'
    
    print(f"Sports: {sports or 'All'}")
    print(f"Nations: {nations or 'All'}")
    print(f"Leagues: {leagues or 'All'}")
    
    # Restart dashboard
    print("\nRestarting dashboard...")
    subprocess.run(['pkill', '-f', 'web_dashboard_real_odds.py'])
    subprocess.Popen(['./restart_dashboard_english_soccer.sh'])
    
    print("\n✅ Dashboard restarted with new filters!")
    print("Access at: http://localhost:8888")

def custom_filters():
    """Get custom filter inputs from user"""
    print("\n=== Custom Filter Setup ===")
    print("Enter comma-separated values (or press Enter for no filter)\n")
    
    sports = input("Sports (e.g., Soccer,Basketball): ").strip()
    nations = input("Nations (e.g., England,Spain,Italy): ").strip()
    leagues = input("Leagues (e.g., Premier League,La Liga): ").strip()
    
    return sports, nations, leagues

def main():
    while True:
        show_menu()
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == 's':
            get_current_filters()
        elif choice == 'c':
            sports, nations, leagues = custom_filters()
            confirm = input("\nApply these filters? (y/n): ").strip().lower()
            if confirm == 'y':
                apply_filters(sports, nations, leagues)
        elif choice in FILTER_PRESETS:
            preset = FILTER_PRESETS[choice]
            print(f"\nSelected: {preset['name']}")
            confirm = input("Apply this preset? (y/n): ").strip().lower()
            if confirm == 'y':
                apply_filters(preset['sports'], preset['nations'], preset['leagues'])
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()