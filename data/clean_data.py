import pandas as pd
import numpy as np
import os
import re

# Resolve absolute path to data folder
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_data():
    raw_path = os.path.join(DATA_DIR, 'crime_data.csv')
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return
        
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Standardize formats
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # Map severities to purely numeric weight column for backend risk score calculation
    # "Low" = 1, "Medium" = 2.5, "High" = 5
    def map_severity(severity_str):
        if pd.isna(severity_str):
            return 1.0
        s = severity_str.lower()
        if 'high' in s: return 3.0
        if 'medium' in s: return 2.0
        return 1.0
        
    df['Severity_Weight'] = df['Severity'].apply(map_severity)
    
    # Drop rows with critical nulls (lat, long, date)
    df.dropna(subset=['Latitude', 'Longitude', 'Date'], inplace=True)
    
    # Write optimized output
    out_path = os.path.join(DATA_DIR, 'cleaned_crime_data.csv')
    df.to_csv(out_path, index=False)
    
    print(f"Cleaned dataset saved successfully to {out_path} [{len(df)} rows].")

if __name__ == '__main__':
    clean_data()
