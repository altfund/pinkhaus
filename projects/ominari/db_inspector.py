#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 20:50:50 2025

@author: ess
"""

import sqlite3
import pandas as pd

def export_query_results(db_name: str, query: str, output_file: str, params=()):
    """
    Executes a SQL query and saves the results as a CSV file.
    """
    try:
        with sqlite3.connect(db_name) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error executing query: {e}")

# Database name
DATABASE_NAME = "sport_odds.db"

# General database structure queries
queries = {
    "list_tables": "SELECT name FROM sqlite_master WHERE type='table';",
    "schema_all_tables": "SELECT name, sql FROM sqlite_master WHERE type='table';"
}

# Execute structure queries
for name, query in queries.items():
    output_file = f"{name}.csv"
    export_query_results(DATABASE_NAME, query, output_file)

# Connect to fetch aggregated database insights
try:
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        
        # Get a list of all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0].strip() for row in cursor.fetchall()]  # Ensure no trailing spaces or characters
        
        aggregated_data = []
        
        for table in tables:
            print(f"Processing table: {table}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            row_count = cursor.fetchone()[0]
            
            # Get column info
            col_info = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            col_names = ", ".join(col_info['name'].tolist()) if not col_info.empty else "No columns found"
            
            # Get a sample row
            sample_data = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1;", conn)
            sample_values = sample_data.to_dict(orient='records')[0] if not sample_data.empty else "No sample data"
            
            aggregated_data.append({
                "Table Name": table,
                "Row Count": row_count,
                "Columns": col_names,
                "Sample Data": sample_values
            })
        
        # Save aggregated data to CSV
        df_aggregated = pd.DataFrame(aggregated_data)
        df_aggregated.to_csv("database_summary.csv", index=False)
        print("Results saved to database_summary.csv")

except Exception as e:
    print(f"Error fetching table details: {e}")

print("All queries executed and results saved.")

