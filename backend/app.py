from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'unified_crime_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml', 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'ml', 'label_encoder.pkl')

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

@app.route('/api/predict', methods=['POST'])
def predict_crime():
    req = request.json
    if not req or 'Latitude' not in req or 'Longitude' not in req:
        return jsonify({"error": "Missing parameters"}), 400
    
    import datetime
    now = datetime.datetime.now()
    
    lat = float(req['Latitude'])
    lng = float(req['Longitude'])
    hour = int(req.get('Hour', now.hour))
    
    if model:
        input_data = pd.DataFrame([[lat, lng, hour]], 
                                  columns=['Latitude', 'Longitude', 'Hour'])
        prediction_num = model.predict(input_data)[0]
        crime_prob = model.predict_proba(input_data)[0]
        
        predicted_type = encoder.inverse_transform([prediction_num])[0]
        max_prob = max(crime_prob)
        
        # Top 3 predicted crimes
        top_indices = np.argsort(crime_prob)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "crime": encoder.inverse_transform([idx])[0],
                "probability": round(float(crime_prob[idx]) * 100, 1)
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
        
        return jsonify({
            "predicted_crime": predicted_type,
            "probability": round(max_prob * 100, 2),
            "top_predictions": top_predictions,
            "safety_score": safety,
            "safety_level": safety_level
        })
    return jsonify({"error": "Model not loaded"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
