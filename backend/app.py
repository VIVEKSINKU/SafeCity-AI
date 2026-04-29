from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import threading
from functools import lru_cache
from dotenv import load_dotenv
import google.generativeai as genai
from geopy.geocoders import Nominatim

# ─── Feature engineering (must match train_model.py exactly) ────────────────
CITY_LAT, CITY_LNG = 31.3260, 75.5762

FEATURES = [
    'Latitude', 'Longitude',
    'Hour', 'DayOfWeek', 'Month',
    'Is_Weekend', 'Is_Night', 'Is_Evening', 'Is_Morning',
    'Hour_sin', 'Hour_cos',
    'Month_sin', 'Month_cos',
    'Day_sin', 'Day_cos',
    'Dist_Centre',
    'Lat_Zone', 'Lng_Zone',
]

# Precomputed lat/lng bin edges (match training data range)
_LAT_BINS = np.linspace(30.9, 31.6, 7)
_LNG_BINS = np.linspace(74.8, 76.0, 7)

def engineer_features(lat: float, lng: float, hour: int,
                      day_of_week: int, month: int) -> pd.DataFrame:
    """Build the same 18-feature row that was used during training."""
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night   = 1 if (hour >= 22 or hour <= 5) else 0
    is_evening = 1 if (18 <= hour < 22) else 0
    is_morning = 1 if (6  <= hour < 12) else 0

    hour_sin  = np.sin(2 * np.pi * hour       / 24)
    hour_cos  = np.cos(2 * np.pi * hour       / 24)
    month_sin = np.sin(2 * np.pi * month      / 12)
    month_cos = np.cos(2 * np.pi * month      / 12)
    day_sin   = np.sin(2 * np.pi * day_of_week / 7)
    day_cos   = np.cos(2 * np.pi * day_of_week / 7)

    dist_centre = np.sqrt((lat - CITY_LAT)**2 + (lng - CITY_LNG)**2)

    lat_zone = int(np.clip(np.digitize(lat, _LAT_BINS) - 1, 0, 5))
    lng_zone = int(np.clip(np.digitize(lng, _LNG_BINS) - 1, 0, 5))

    row = {
        'Latitude': lat, 'Longitude': lng,
        'Hour': hour, 'DayOfWeek': day_of_week, 'Month': month,
        'Is_Weekend': is_weekend, 'Is_Night': is_night,
        'Is_Evening': is_evening, 'Is_Morning': is_morning,
        'Hour_sin': hour_sin, 'Hour_cos': hour_cos,
        'Month_sin': month_sin, 'Month_cos': month_cos,
        'Day_sin': day_sin, 'Day_cos': day_cos,
        'Dist_Centre': dist_centre,
        'Lat_Zone': lat_zone, 'Lng_Zone': lng_zone,
    }
    return pd.DataFrame([row])[FEATURES]

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'unified_crime_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml', 'xgb_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'ml', 'label_encoder.pkl')

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

geolocator = Nominatim(user_agent="safecity_ai")

# ─── Load data and models once at startup ───────────────────────────
try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print(f"Loaded {len(df)} records, {df['Crime_Type'].nunique()} crime categories")
except Exception as e:
    print(f"Error loading models or data: {e}")
    df = pd.DataFrame()
    model = None
    encoder = None

# ─── Pre-compute hotspots at startup (cached) ──────────────────────
CACHED_HOTSPOTS = []

def compute_hotspots():
    global CACHED_HOTSPOTS
    if df.empty:
        return
    try:
        from sklearn.cluster import DBSCAN
        coords = df[['Latitude', 'Longitude']].values
        db = DBSCAN(eps=0.008, min_samples=10).fit(coords)
        df['Cluster'] = db.labels_
        clusters = df[df['Cluster'] != -1]
        
        hotspot_data = []
        if not clusters.empty:
            for cluster_id, group in clusters.groupby('Cluster'):
                lat_center = group['Latitude'].mean()
                lng_center = group['Longitude'].mean()
                incident_count = len(group)
                
                primary_crime = group['Crime_Type'].mode()[0] if not group.empty else "Unknown"
                
                # Top 3 crimes for richer analytics
                crime_counts = group['Crime_Type'].value_counts().head(3)
                top_crimes = [{"type": ct, "count": int(cc)} for ct, cc in crime_counts.items()]
                
                rush_hour = int(group['Hour'].mode()[0]) if 'Hour' in group.columns and not group.empty else 12
                rush_hour_str = f"{rush_hour:02d}:00 - {(rush_hour+1)%24:02d}:00"
                
                # Hour distribution for the cluster
                hour_dist = group['Hour'].value_counts().sort_index().to_dict()
                peak_hours = sorted(hour_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                
                location_name = group['Location'].mode()[0] if 'Location' in group.columns and not group.empty else "Unknown Region"
                
                # Severity breakdown
                sev_counts = group['Severity'].value_counts().to_dict()
                
                weight = min(1.0, max(0.2, incident_count / 15))
                
                hotspot_data.append({
                    "lat": float(lat_center),
                    "lng": float(lng_center),
                    "weight": float(weight),
                    "cluster_size": int(incident_count),
                    "primary_crime": primary_crime,
                    "top_crimes": top_crimes,
                    "rush_hour": rush_hour_str,
                    "peak_hours": [{"hour": f"{int(h):02d}:00", "count": int(c)} for h, c in peak_hours],
                    "location_name": location_name,
                    "severity": sev_counts
                })
        
        hotspot_data.sort(key=lambda x: x['weight'], reverse=True)
        CACHED_HOTSPOTS = hotspot_data[:15]
        print(f"Pre-computed {len(CACHED_HOTSPOTS)} hotspot clusters")
    except Exception as e:
        print(f"Clustering error: {e}")
        CACHED_HOTSPOTS = [{"lat": 31.3260, "lng": 75.5762, "weight": 0.8, 
                            "cluster_size": 0, "primary_crime": "Unknown",
                            "top_crimes": [], "rush_hour": "N/A", "peak_hours": [],
                            "location_name": "Jalandhar", "severity": {}}]

compute_hotspots()

# ─── Severity weights for safety score calculation ──────────────────
SEVERITY_WEIGHTS = {
    'Violent Crime': 10,
    'Sexual Crime': 10,
    'Terrorism': 9,
    'Kidnapping': 8,
    'Assault': 7,
    'Drug Crime': 5,
    'Property Crime': 3,
    'Cybercrime': 2,
    'Petty Crime': 4
}

def calculate_safety_score(lat, lng, hour, prediction_probs, class_names):
    """
    Compute a 0-100 safety score. 100 = perfectly safe, 0 = extremely dangerous.
    Factors: predicted crime probabilities, proximity to hotspots, time of day.
    """
    # 1. Crime probability danger (weighted by severity)
    weighted_danger = 0
    for prob, crime_name in zip(prediction_probs, class_names):
        weighted_danger += prob * SEVERITY_WEIGHTS.get(crime_name, 4)
    # Normalize: max possible is 10 (if 100% of worst crime)
    crime_score = (weighted_danger / 10) * 40  # contributes up to 40 points of danger
    
    # 2. Proximity to hotspots
    hotspot_danger = 0
    for hs in CACHED_HOTSPOTS:
        dist = np.sqrt((lat - hs['lat'])**2 + (lng - hs['lng'])**2)
        if dist < 0.02:  # ~2km
            hotspot_danger = max(hotspot_danger, hs['weight'] * 30)
    
    # 3. Time of day risk (late night = more dangerous)
    night_hours = {0: 15, 1: 18, 2: 20, 3: 20, 4: 18, 5: 12, 
                   22: 12, 23: 15}
    time_danger = night_hours.get(hour, 5)
    
    total_danger = min(95, crime_score + hotspot_danger + time_danger)
    safety_score = max(5, round(100 - total_danger))
    
    return safety_score

# ─── Routes ─────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "message": "Jalandhar Sentinel AI Backend is Running",
        "endpoints": ["/api/historical-data", "/api/hotspots", "/api/predict"]
    })

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
    
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    cols = ['Latitude', 'Longitude', 'Crime_Type', 'Severity']
    if 'Hour' in df.columns:
        cols.append('Hour')
    if 'Location' in df.columns:
        cols.append('Location')
        
    data = sample_df[cols].to_dict(orient='records')
    return jsonify(data)

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    return jsonify(CACHED_HOTSPOTS)

# ─── Cached geocoding (single source of truth) ──────────────────────
_geocode_lock = threading.Lock()
_geocode_cache = {}        # (rounded_lat, rounded_lng) -> location_name
_PRECISION = 4             # 4 decimal places ≈ 11m accuracy

def _round_coord(lat, lng):
    """Round coordinates to cache-friendly precision."""
    return round(lat, _PRECISION), round(lng, _PRECISION)

def _reverse_geocode_raw(lat, lng):
    """Nominatim-only reverse geocode → structured name (no Gemini)."""
    try:
        location = geolocator.reverse((lat, lng), exactly_one=True, timeout=3)
        if location:
            address = location.raw.get('address', {})
            road   = address.get('road', address.get('street', ''))
            suburb = address.get('suburb', address.get('neighbourhood', address.get('residential', '')))
            city   = address.get('city', address.get('town', address.get('village', '')))
            parts  = [p for p in [road, suburb, city] if p]
            if parts:
                return ", ".join(parts), location.address
            return location.address.split(',')[0], location.address
    except Exception as e:
        print(f"Nominatim error: {e}")
    return "Unknown Location", ""

def _format_with_gemini(raw_name, full_address):
    """Use Gemini to clean-format the address (optional enhancement)."""
    if not gemini_model or raw_name == "Unknown Location":
        return raw_name
    try:
        prompt = (
            f"Format this raw geographic address into a clean "
            f"'Road/Street, Neighborhood, City' format "
            f"(e.g. 'Sodal Mandir Road, Preet Nagar, Jalandhar'): "
            f"'{full_address or raw_name}'. Return ONLY the cleaned name."
        )
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini format error: {e}")
        return raw_name

def get_location_name(lat, lng, use_gemini=True):
    """
    Cached geocoding lookup.  Nearby coordinates (within ~11m) share a
    single cached result, so rapid map clicks don't spam external APIs.
    """
    key = _round_coord(lat, lng)
    with _geocode_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]

    raw_name, full_addr = _reverse_geocode_raw(lat, lng)
    if use_gemini:
        final_name = _format_with_gemini(raw_name, full_addr)
    else:
        final_name = raw_name

    with _geocode_lock:
        _geocode_cache[key] = final_name
        # Cap cache at 500 entries (LRU-style eviction)
        if len(_geocode_cache) > 500:
            oldest = next(iter(_geocode_cache))
            del _geocode_cache[oldest]

    return final_name

@app.route('/api/geocode', methods=['POST'])
def geocode_location():
    req = request.json
    if not req or 'Latitude' not in req or 'Longitude' not in req:
        return jsonify({"error": "Missing parameters"}), 400

    lat = float(req['Latitude'])
    lng = float(req['Longitude'])
    location_name = get_location_name(lat, lng, use_gemini=True)
    return jsonify({"location_name": location_name})

@app.route('/api/predict', methods=['POST'])
def predict_crime():
    req = request.json
    if not req or 'Latitude' not in req or 'Longitude' not in req:
        return jsonify({"error": "Missing parameters"}), 400
    
    now = datetime.datetime.now()
    
    lat  = float(req['Latitude'])
    lng  = float(req['Longitude'])
    hour        = int(req.get('Hour',      now.hour))
    day_of_week = int(req.get('DayOfWeek', now.weekday()))   # 0=Mon … 6=Sun
    month       = int(req.get('Month',     now.month))
    
    if model:
        input_data   = engineer_features(lat, lng, hour, day_of_week, month)
        prediction_num = model.predict(input_data)[0]
        crime_prob     = model.predict_proba(input_data)[0]
        
        raw_predicted_type = encoder.inverse_transform([prediction_num])[0]
        predicted_type = "Petty Crime" if raw_predicted_type.lower() == "other" else raw_predicted_type
        max_prob = max(crime_prob)
        
        # Top 3 predicted crimes
        top_indices = np.argsort(crime_prob)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            raw_crime_name = encoder.inverse_transform([idx])[0]
            crime_name = "Petty Crime" if raw_crime_name.lower() == "other" else raw_crime_name
            top_predictions.append({
                "crime": crime_name,
                "probability": float(round(float(crime_prob[idx]) * 100, 1))
            })
        
        # Safety score
        safety = calculate_safety_score(lat, lng, hour, crime_prob, encoder.classes_)
        
        # Safety level label
        if safety >= 70:
            safety_level = "Safe"
        elif safety >= 40:
            safety_level = "Caution"
        else:
            safety_level = "Dangerous"
        
        # Reuse the cached geocoding (already resolved by frontend click)
        # Accept pre-resolved name from frontend to avoid a second lookup.
        location_name = req.get('location_name', '')
        if not location_name:
            # Fallback: use cached lookup (will be instant if user already clicked)
            location_name = get_location_name(lat, lng, use_gemini=False)
        
        # Single Gemini call: safety insight only (no redundant formatting)
        gemini_insight = "AI insight unavailable."
        if gemini_model:
            try:
                prompt = (
                    f"The user is at {location_name} (Lat: {lat}, Lng: {lng}) "
                    f"at hour {hour}:00. The calculated safety score is {safety}/100 "
                    f"and the highest predicted risk is {predicted_type}. "
                    f"Write a short, 2-sentence actionable safety recommendation "
                    f"for the user. Keep it natural and do not use bold formatting."
                )
                response = gemini_model.generate_content(prompt)
                gemini_insight = response.text.strip()
            except Exception as e:
                print(f"Gemini insight error: {e}")
        
        return jsonify({
            "predicted_crime": str(predicted_type),
            "probability": float(round(max_prob * 100, 2)),
            "top_predictions": top_predictions,
            "safety_score": int(safety),
            "safety_level": str(safety_level),
            "location_name": str(location_name),
            "gemini_insight": str(gemini_insight)
        })
    return jsonify({"error": "Model not loaded"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
