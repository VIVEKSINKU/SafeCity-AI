import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Jalandhar Bounding Box roughly
LAT_MIN = 31.25
LAT_MAX = 31.40
LON_MIN = 75.50
LON_MAX = 75.65

CRIME_TYPES = ['Theft', 'Assault', 'Burglary', 'Vandalism', 'Cybercrime']
SEVERITY_LEVELS = ['Low', 'Medium', 'High']

# Centers of specific "hotspots" to bias the data generation
HOTSPOTS = [
    (31.3260, 75.5762), # City Center
    (31.3114, 75.5866), # Model Town
    (31.3414, 75.5866)  # Industrial Area (example)
]

def generate_random_date(start_year=2021, end_year=2024):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1) - timedelta(days=1)
    random_days = random.randint(0, (end_date - start_date).days)
    random_time = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return start_date + timedelta(days=random_days) + random_time

def generate_data(num_records=5000):
    data = []
    
    for _ in range(num_records):
        # 60% chance to be near a hotspot
        if random.random() < 0.6:
            hotspot = random.choice(HOTSPOTS)
            lat = np.random.normal(hotspot[0], 0.01)
            lon = np.random.normal(hotspot[1], 0.01)
        else:
            lat = random.uniform(LAT_MIN, LAT_MAX)
            lon = random.uniform(LON_MIN, LON_MAX)
            
        dt = generate_random_date()
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H:%M:%S')
        
        crime_type = random.choice(CRIME_TYPES)
        # Assign severity somewhat based on crime type
        if crime_type in ['Assault', 'Cybercrime']:
            severity = random.choices(SEVERITY_LEVELS, weights=[0.2, 0.4, 0.4])[0]
        else:
            severity = random.choices(SEVERITY_LEVELS, weights=[0.5, 0.3, 0.2])[0]
            
        data.append({
            'Date': date_str,
            'Time': time_str,
            'Latitude': lat,
            'Longitude': lon,
            'Crime_Type': crime_type,
            'Severity': severity
        })
        
    df = pd.DataFrame(data)
    df.to_csv('crime_data.csv', index=False)
    print(f"Successfully generated {num_records} records in crime_data.csv")

if __name__ == "__main__":
    generate_data()
