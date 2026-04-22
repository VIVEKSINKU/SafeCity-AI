import React, { useState } from 'react';
import axios from 'axios';
import MapComponent from './components/MapComponent';
import { ShieldAlert, Crosshair, Map as MapIcon, Clock, Activity, AlertTriangle, Shield, ChevronDown } from 'lucide-react';

const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

function SafetyGauge({ score, level }) {
  const getColor = () => {
    if (score >= 70) return { bar: 'from-green-400 to-emerald-500', text: 'text-green-400', bg: 'bg-green-900/20', border: 'border-green-500/30' };
    if (score >= 40) return { bar: 'from-yellow-400 to-orange-500', text: 'text-yellow-400', bg: 'bg-yellow-900/20', border: 'border-yellow-500/30' };
    return { bar: 'from-red-400 to-red-600', text: 'text-red-400', bg: 'bg-red-900/20', border: 'border-red-500/30' };
  };
  const colors = getColor();
  const Icon = score >= 70 ? Shield : AlertTriangle;

  return (
    <div className={`p-4 rounded-xl ${colors.bg} border ${colors.border} mb-4`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${colors.text}`} />
          <span className={`text-sm font-semibold ${colors.text}`}>Safety Score</span>
        </div>
        <span className={`text-3xl font-bold ${colors.text}`}>{score}</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-3 mb-2">
        <div className={`bg-gradient-to-r ${colors.bar} h-3 rounded-full transition-all duration-700`} style={{ width: `${score}%` }}></div>
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>Dangerous</span>
        <span className={`font-semibold ${colors.text}`}>{level}</span>
        <span>Safe</span>
      </div>
    </div>
  );
}

function App() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedHour, setSelectedHour] = useState(new Date().getHours());
  const [selectedDay, setSelectedDay] = useState(new Date().getDay());

  const handlePredict = async () => {
    if (!selectedLocation) {
      alert("Please click a location on the map first.");
      return;
    }

    setLoading(true);
    try {
      const payload = {
        Latitude: selectedLocation.lat,
        Longitude: selectedLocation.lng,
        Hour: selectedHour,
        DayOfWeek: selectedDay,
        Month: new Date().getMonth() + 1
      };

      const res = await axios.post('http://127.0.0.1:5000/api/predict', payload);
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans overflow-hidden flex flex-col">
      {/* Navbar */}
      <header className="px-8 py-4 glass-panel border-t-0 border-r-0 border-l-0 rounded-none flex items-center justify-between z-10 relative">
        <div className="flex items-center gap-3">
          <ShieldAlert className="text-red-500 w-8 h-8" />
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-orange-500">
            SafeCity AI
          </h1>
        </div>
        <div className="text-sm text-slate-300 flex items-center gap-2">
          <Activity className="text-green-400 w-4 h-4 ml-2" /> Live System Status: Active
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow p-6 flex flex-col lg:flex-row gap-6 h-[calc(100vh-80px)]">

        {/* Left Sidebar (Controls) */}
        <div className="w-full lg:w-1/3 flex flex-col gap-4 overflow-y-auto">

          <div className="glass-panel p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Crosshair className="text-blue-400" /> AI Prediction Engine
            </h2>
            <p className="text-slate-300 text-sm mb-4">
              Select a location on the map, choose the time scenario, and run the risk analysis.
            </p>

            {/* Location Display */}
            <div className="bg-slate-800/50 p-3 rounded-lg mb-4 border border-slate-700">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-slate-400">Target Location</span>
                <MapIcon className="w-3 h-3 text-slate-500" />
              </div>
              <div className="font-mono text-sm text-blue-300">
                {selectedLocation
                  ? `${selectedLocation.lat.toFixed(4)}° N, ${selectedLocation.lng.toFixed(4)}° E`
                  : "Click on the map to select"}
              </div>
            </div>

            {/* Time & Day Selectors */}
            <div className="flex gap-3 mb-4">
              <div className="flex-1">
                <label className="text-xs text-slate-400 mb-1 block">Time of Day</label>
                <select
                  value={selectedHour}
                  onChange={(e) => setSelectedHour(parseInt(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 appearance-none cursor-pointer"
                >
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>
                      {i === 0 ? '12:00 AM' : i < 12 ? `${i}:00 AM` : i === 12 ? '12:00 PM' : `${i - 12}:00 PM`}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex-1">
                <label className="text-xs text-slate-400 mb-1 block">Day of Week</label>
                <select
                  value={selectedDay}
                  onChange={(e) => setSelectedDay(parseInt(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 appearance-none cursor-pointer"
                >
                  {DAY_NAMES.map((name, i) => (
                    <option key={i} value={i}>{name}</option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !selectedLocation}
              className={`w-full py-3 rounded-xl font-medium transition-all ${loading || !selectedLocation
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-900/50'
                }`}
            >
              {loading ? 'Analyzing Threat Matrix...' : 'Run Risk Analysis'}
            </button>
          </div>

          {/* Prediction Result Box */}
          {prediction && (
            <div className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-blue-300 mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-400" /> Threat Assessment
              </h3>

              {/* Safety Score Gauge */}
              <SafetyGauge score={prediction.safety_score} level={prediction.safety_level} />

              {/* Top Predictions */}
              <div className="space-y-2">
                <div className="text-xs text-slate-400 mb-1">Predicted Crime Probabilities</div>
                {prediction.top_predictions && prediction.top_predictions.map((tp, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="flex justify-between text-xs mb-1">
                        <span className={`capitalize ${i === 0 ? 'text-red-400 font-semibold' : 'text-slate-400'}`}>
                          {tp.crime}
                        </span>
                        <span className="text-slate-500">{tp.probability}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full ${i === 0 ? 'bg-red-500' : i === 1 ? 'bg-orange-500' : 'bg-yellow-500'}`}
                          style={{ width: `${tp.probability}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Right Area (Map) */}
        <div className="w-full lg:w-2/3 h-[50vh] lg:h-full glass-panel relative p-1">
          <MapComponent onLocationSelect={setSelectedLocation} />
        </div>

      </main>
    </div>
  );
}

export default App;
