import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import 'leaflet.heat';
import axios from 'axios';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

function LocationMarker({ onLocationSelect }) {
  const [position, setPosition] = useState(null)
  useMapEvents({
    click(e) {
      setPosition(e.latlng)
      onLocationSelect(e.latlng)
    },
  })
  return position === null ? null : (
    <Marker position={position}>
      <Popup>Selected Location for Prediction</Popup>
    </Marker>
  )
}

function HeatmapLayer({ historicalData }) {
  const map = useMap();

  useEffect(() => {
    if (!map || historicalData.length === 0) return;

    const points = historicalData.map(pt => {
      let intensity = 0.5;
      if (pt.Severity === 'High') intensity = 1.0;
      if (pt.Severity === 'Medium') intensity = 0.75;
      return [pt.Latitude, pt.Longitude, intensity];
    });

    const heatLayer = L.heatLayer(points, {
      radius: 35,
      blur: 25,
      maxZoom: 15,
      max: 1.0,
      gradient: { 0.2: '#2563eb', 0.4: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1: '#ef4444' }
    }).addTo(map);

    return () => {
      map.removeLayer(heatLayer);
    };
  }, [map, historicalData]);

  return null;
}

const MapComponent = ({ onLocationSelect }) => {
  const [hotspots, setHotspots] = useState([]);
  const [historical, setHistorical] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/hotspots')
      .then(res => setHotspots(res.data))
      .catch(err => console.error("Error fetching hotspots", err));

    axios.get('http://127.0.0.1:5000/api/historical-data')
      .then(res => setHistorical(res.data))
      .catch(err => console.error("Error fetching historical data", err));
  }, []);

  const mapCenter = [31.2900, 75.6400];

  return (
    <MapContainer center={mapCenter} zoom={12} className="h-full w-full rounded-2xl shadow-inner z-0">
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />

      <HeatmapLayer historicalData={historical} />

      {/* Clickable Hotspot Overlays with Danger Analytics */}
      {hotspots.map((hs, idx) => (
        <Circle
          key={`hs-${idx}`}
          center={[hs.lat, hs.lng]}
          pathOptions={{ 
            color: hs.weight > 0.7 ? '#ef4444' : '#f59e0b', 
            fillColor: hs.weight > 0.7 ? '#ef4444' : '#f59e0b', 
            fillOpacity: 0.08,
            weight: 1,
            dashArray: '5 5'
          }}
          radius={600 * hs.weight}
        >
          <Popup>
            <div style={{ minWidth: '220px', fontFamily: 'system-ui', color: '#1e293b' }}>
              <h3 style={{ margin: '0 0 8px', fontSize: '15px', fontWeight: 700, borderBottom: '2px solid #ef4444', paddingBottom: '6px' }}>
                Danger Zone Analytics
              </h3>
              <p style={{ margin: '4px 0', fontSize: '13px' }}>
                <strong>Region:</strong> {hs.location_name}
              </p>
              <p style={{ margin: '4px 0', fontSize: '13px' }}>
                <strong>Top Crime:</strong> <span style={{ color: '#dc2626' }}>{hs.primary_crime}</span>
              </p>
              {hs.top_crimes && hs.top_crimes.length > 1 && (
                <div style={{ margin: '6px 0', fontSize: '12px', background: '#f1f5f9', padding: '6px 8px', borderRadius: '6px' }}>
                  {hs.top_crimes.map((tc, i) => (
                    <div key={i}>{tc.type}: {tc.count} incidents</div>
                  ))}
                </div>
              )}
              <p style={{ margin: '4px 0', fontSize: '13px' }}>
                <strong>Peak Hour:</strong> {hs.rush_hour}
              </p>
              <p style={{ margin: '6px 0 0', fontSize: '11px', color: '#94a3b8', borderTop: '1px solid #e2e8f0', paddingTop: '6px' }}>
                Aggregated from {hs.cluster_size} incidents
              </p>
            </div>
          </Popup>
        </Circle>
      ))}

      <LocationMarker onLocationSelect={onLocationSelect} />
    </MapContainer>
  );
};

export default MapComponent;
