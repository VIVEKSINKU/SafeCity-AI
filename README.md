# SafeCity-AI: Intelligent Crime Prediction and Prevention

A comprehensive crime mapping and prediction system that uses machine learning to forecast crime hotspots and provides real-time safety intelligence to users.

## Features

- **Crime Prediction**: Predicts crime types with probabilities for any given location and time.
- **Hotspot Detection**: Identifies high-risk areas using DBSCAN clustering and visualizes them on a map.
- **Safety Score**: Real-time safety assessment for user-selected locations.
- **AI-Powered Insights**: Generates intelligent, location-specific safety tips using Gemini.
- **Geocoding**: Automatic location naming and reverse geocoding.
- **Time-Aware Prediction**: Considers temporal factors (hour, day, month) for higher accuracy.

## Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd crimemapping
    ```

2.  **Install Backend Dependencies:**
    Navigate to the backend directory and install required packages.
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the `backend` directory:
    ```bash
    touch .env
    ```
    Add your Gemini API key to the `.env` file:
    ```env
    GEMINI_API_KEY=your-gemini-api-key-here
    ```

4.  **Run the Server:**
    Start the Flask server:
    ```bash
    python app.py
    ```
    The server will start at `http://localhost:5000`.

5.  **Open the Frontend:**
    Open `frontend/index.html` in your web browser.
    ```bash
    # Navigate to frontend directory
    cd ../frontend
    
    # Open index.html in browser
    # On Windows:
    start index.html
    ```

## ⚙️ Configuration

### Backend Configuration

Edit `backend/.env` to configure:
- `GEMINI_API_KEY`: Your Google Gemini API key for AI insights.
- `FLASK_ENV`: Set to `development` or `production`.
- `PORT`: The port to run the server on (default: 5000).

## 🛠️ Development

### Training the Model

To retrain the machine learning model with new data:

1.  Ensure your dataset is in `backend/data/unified_crime_data.csv`.
2.  Run the training script:
    ```bash
    cd backend
    python train_model.py
    ```

### Troubleshooting

- **Server not starting**: Ensure no other process is using port 5000 or update the `PORT` in `.env`.
- **Gemini API errors**: Check your API key in `.env` and ensure you have billing enabled.
- **Model not found**: Run `train_model.py` to generate the required `.pkl` files.