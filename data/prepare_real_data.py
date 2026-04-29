import pandas as pd
import glob
import os
import numpy as np
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Crime Type Normalization Map ───────────────────────────────────────
# Maps raw messy labels into 8 clean, trainable categories
CRIME_CATEGORY_KEYWORDS = {
    'Violent Crime': [
        'murder', 'homicide', 'killing', 'assassination', 'shot dead',
        'stabbing', 'stab', 'beat to death', 'honour killing', 'honor killing',
        'lynching', 'custodial death'
    ],
    'Sexual Crime': [
        'rape', 'gangrape', 'gang rape', 'sexual assault', 'sexual harassment',
        'molestation', 'eve-teasing', 'eve teasing', 'stalking', 'voyeurism',
        'obscene', 'indecent'
    ],
    'Property Crime': [
        'theft', 'snatching', 'robbery', 'burglary', 'loot', 'pickpocket',
        'shoplifting', 'break-in', 'house breaking', 'vehicle theft',
        'mobile snatching', 'chain snatching', 'dacoity'
    ],
    'Drug Crime': [
        'drug', 'heroin', 'narcotic', 'cocaine', 'opium', 'smack',
        'chitta', 'poppy', 'ganja', 'cannabis', 'mdma', 'contraband'
    ],
    'Terrorism': [
        'terrorism', 'extremism', 'terrorist', 'bomb', 'blast', 'ied',
        'militant', 'separatist', 'khalistani', 'explosives'
    ],
    'Kidnapping': [
        'kidnap', 'abduction', 'hostage', 'ransom', 'missing person'
    ],
    'Cybercrime': [
        'cyber', 'online fraud', 'digital arrest', 'phishing', 'hacking',
        'identity theft', 'upi fraud', 'olx fraud', 'internet'
    ],
    'Assault': [
        'assault', 'attack', 'firing', 'shootout', 'fight', 'brawl',
        'grievous hurt', 'acid attack', 'attempt to murder', 'armed'
    ]
}

# ─── Severity by Category ────────────────────────────────────────────
SEVERITY_MAP = {
    'Violent Crime': 'High',
    'Sexual Crime': 'High',
    'Terrorism': 'High',
    'Kidnapping': 'High',
    'Assault': 'Medium',
    'Drug Crime': 'Medium',
    'Property Crime': 'Low',
    'Cybercrime': 'Low',
    'petty Crime': 'Low'
}

# ─── Location Coordinate Mappings ────────────────────────────────────
# More granular than just 4 buckets
LOCATION_COORDS = {
    # LPU & surrounding
    'lpu': (31.2558, 75.7051),
    'lovely professional university': (31.2558, 75.7051),
    'law gate': (31.2590, 75.7010),
    
    # Phagwara
    'phagwara': (31.2240, 75.7708),
    'kapurthala': (31.3808, 75.3803),
    
    # Jalandhar neighborhoods
    'model town': (31.3114, 75.5866),
    'avtar nagar': (31.3300, 75.5650),
    'basti sheikh': (31.3290, 75.5720),
    'nakodar': (31.1253, 75.4735),
    'shahkot': (31.0814, 75.3325),
    'goraya': (31.1523, 75.7702),
    'adampur': (31.4325, 75.7165),
    'kartarpur': (31.4396, 75.4983),
    'jalandhar cantonment': (31.2972, 75.6107),
    'jalandhar city': (31.3260, 75.5762),
    'jalandhar': (31.3260, 75.5762),
    
    # Wider Punjab locations that appear in the data
    'ludhiana': (30.9010, 75.8573),
    'amritsar': (31.6340, 74.8723),
    'mohali': (30.7046, 76.7179),
    'patiala': (30.3398, 76.3869),
    'bathinda': (30.2110, 74.9455),
    'moga': (30.8163, 75.1742),
    'ferozepur': (30.9255, 74.6132),
    'hoshiarpur': (31.5314, 75.9115),
    'gurdaspur': (32.0414, 75.4027),
    'doaba': (31.3260, 75.6500),
    'punjab': (31.1471, 75.3412),
}

def normalize_crime_type(raw_crime):
    """Map a raw crime description to one of 8 broad categories."""
    raw_lower = str(raw_crime).lower()
    for category, keywords in CRIME_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in raw_lower:
                return category
    return 'Petty Crime'

def get_base_coord(location):
    """Map a location string to coordinates using keyword matching."""
    loc_lower = str(location).lower()
    # Try each mapped location from most specific to least
    for key, coords in LOCATION_COORDS.items():
        if key in loc_lower:
            return coords
    # Default fallback
    return (31.3260, 75.5762)

def parse_time_to_hour(time_str):
    """Extract hour from various time formats."""
    if pd.isna(time_str):
        return np.random.randint(0, 24)
    time_str = str(time_str).strip()
    match = re.search(r'(\d{1,2}):(\d{2})', time_str)
    if match:
        return int(match.group(1))
    time_lower = time_str.lower()
    if 'morning' in time_lower or 'dawn' in time_lower: return 7
    elif 'afternoon' in time_lower: return 14
    elif 'evening' in time_lower or 'dusk' in time_lower: return 19
    elif 'night' in time_lower: return 22
    elif 'midnight' in time_lower: return 0
    elif 'late night' in time_lower: return 2
    else:
        return np.random.randint(0, 24)

def parse_date_fields(date_str):
    """Try to extract month and day-of-week from date strings."""
    if pd.isna(date_str):
        return None, None
    date_str = str(date_str).strip()
    # Try common formats
    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%B %d, %Y', '%Y']:
        try:
            dt = pd.to_datetime(date_str, format=fmt)
            return dt.month, dt.dayofweek
        except:
            pass
    # Try pandas auto-parsing 
    try:
        dt = pd.to_datetime(date_str, dayfirst=True)
        return dt.month, dt.dayofweek
    except:
        return None, None

def main():
    print("=" * 60)
    print("  SafeCity AI - Data Preparation Pipeline")
    print("=" * 60)
    
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    exclude = ['clean', 'unified', 'crime_data']
    valid_files = [f for f in csv_files if not any(x in os.path.basename(f).lower() for x in exclude)]
    
    print(f"\nFound {len(valid_files)} raw data files.")
    
    unified_records = []
    
    for file in valid_files:
        fname = os.path.basename(file)
        df = pd.read_csv(file)
        
        loc_col = next((c for c in df.columns if 'location' in str(c).lower()), None)
        crime_col = next((c for c in df.columns if 'crime' in str(c).lower() or 'incident' in str(c).lower()), None)
        time_col = next((c for c in df.columns if 'time' in str(c).lower()), None)
        date_col = next((c for c in df.columns if 'date' in str(c).lower()), None)
        
        count = 0
        if loc_col and crime_col:
            for _, row in df.iterrows():
                if not pd.isna(row[loc_col]) and not pd.isna(row[crime_col]):
                    unified_records.append({
                        'Location': row[loc_col],
                        'Raw_Crime': row[crime_col],
                        'Raw_Time': row.get(time_col) if time_col else None,
                        'Raw_Date': row.get(date_col) if date_col else None,
                    })
                    count += 1
        print(f"  [{count:3d} rows] {fname}")
    
    master_df = pd.DataFrame(unified_records)
    print(f"\nTotal extracted records: {len(master_df)}")
    
    # ─── Normalize Crime Types ──────────────────────────────────────
    master_df['Crime_Type'] = master_df['Raw_Crime'].apply(normalize_crime_type)
    
    print("\nNormalized Crime Distribution:")
    for cat, count in master_df['Crime_Type'].value_counts().items():
        print(f"  {cat:20s} : {count}")
    
    # ─── Severity ───────────────────────────────────────────────────
    master_df['Severity'] = master_df['Crime_Type'].map(SEVERITY_MAP)
    
    # ─── Geocode Locations ──────────────────────────────────────────
    lats, lons = [], []
    for _, row in master_df.iterrows():
        base_lat, base_lon = get_base_coord(row['Location'])
        # Scatter ~1.5km so points don't stack
        lats.append(base_lat + np.random.normal(0, 0.012))
        lons.append(base_lon + np.random.normal(0, 0.012))
    master_df['Latitude'] = lats
    master_df['Longitude'] = lons
    
    # ─── Temporal Features ──────────────────────────────────────────
    hours = []
    months = []
    days = []
    for _, row in master_df.iterrows():
        hours.append(parse_time_to_hour(row['Raw_Time']))
        m, d = parse_date_fields(row['Raw_Date'])
        months.append(m if m else np.random.randint(1, 13))
        days.append(d if d is not None else np.random.randint(0, 7))
    
    master_df['Hour'] = hours
    master_df['Month'] = months
    master_df['DayOfWeek'] = days
    
    # ─── Output ─────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, 'unified_crime_data.csv')
    final_cols = ['Latitude', 'Longitude', 'DayOfWeek', 'Month', 'Hour', 'Crime_Type', 'Severity', 'Location']
    master_df[final_cols].to_csv(out_path, index=False)
    
    print(f"\n[OK] Saved {len(master_df)} clean records to {out_path}")
    print(f"   Unique crime categories: {master_df['Crime_Type'].nunique()}")
    print(f"   Severity distribution: {dict(master_df['Severity'].value_counts())}")

if __name__ == '__main__':
    main()
