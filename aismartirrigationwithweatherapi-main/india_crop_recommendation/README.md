# 🌾 India Crop Recommendation System

AI-powered crop recommendation system for Indian agriculture using soil moisture sensors (manual entry), weather data, machine learning, and **RAG (Retrieval-Augmented Generation)** for intelligent document-based Q&A.

## 🆕 RAG Knowledge Base Features

The system now includes a powerful RAG (Retrieval-Augmented Generation) system that enables:

### Document Upload & Processing
- **Supported Formats**: PDF, TXT, CSV, JSON, Markdown
- **Automatic Chunking**: Documents are split into semantic chunks with overlap
- **Embedding Generation**: TF-IDF based embeddings for semantic search
- **Metadata Tagging**: Add state, crop, category, source for filtering

### Hybrid Search
- **Semantic Search**: Find contextually similar content using embeddings
- **Keyword Search (BM25)**: Match exact terms for precise queries
- **Metadata Filtering**: Filter by state, crop, category, date range
- **Hybrid Retrieval**: Combine semantic + keyword for best results

### RAG-Enhanced Chat
- Queries the knowledge base for relevant context
- Augments LLM prompts with retrieved information
- Provides source citations with confidence scores
- Falls back to general knowledge when no relevant documents exist

### API Endpoints
```
POST /rag/upload       - Upload documents to knowledge base
POST /rag/query        - Search with hybrid retrieval
POST /rag/chat         - RAG-enhanced chat with LLM
GET  /rag/documents    - List all documents
DELETE /rag/documents/{id} - Remove a document
GET  /rag/stats        - Knowledge base statistics
```

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDIA CROP RECOMMENDATION SYSTEM                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │   CSV Data   │   │  Weather API │   │  Manual UI   │                     │
│  │ (states.csv) │   │(OpenWeather) │   │ (20 sensors) │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                              │
│         ▼                  ▼                  ▼                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     INGEST LAYER (Python)                          │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │     │
│  │  │weather_api  │  │soil_moisture│  │  crop_data  │                │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    ETL PIPELINE (Airflow DAG)                      │     │
│  │  ingest_weather → ingest_soil → validate → dedupe → load_postgres │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                │                                             │
│         ┌──────────────────────┼──────────────────────┐                     │
│         ▼                      ▼                      ▼                     │
│  ┌─────────────┐       ┌─────────────┐        ┌─────────────┐              │
│  │  Parquet    │       │ PostgreSQL  │        │  ML Models  │              │
│  │  Storage    │       │   (prod)    │        │ (LightGBM)  │              │
│  └─────────────┘       └─────────────┘        └─────────────┘              │
│                                │                      │                     │
│                                ▼                      ▼                     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    REST API (FastAPI)                               │     │
│  │   POST /recommend   GET /status   GET /states   GET /crops         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    WEB UI (React + Tailwind)                        │     │
│  │   📍 Location   💧 20 Sensor Inputs   🌱 Recommendations           │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
india_crop_recommendation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container config
├── docker-compose.yml           # Airflow + Postgres services
│
├── config.py                    # Central configuration (36 states, crops, paths)
├── data_sources.py              # Data source documentation (10 sources)
├── schema.py                    # PostgreSQL DDL + Parquet layouts
│
├── ingest/                      # Data ingestion modules
│   ├── __init__.py
│   ├── weather_api.py           # OpenWeatherMap integration
│   ├── soil_moisture.py         # Satellite + CSV ingestion
│   └── test_weather_api.py      # pytest test suite
│
├── dags/                        # Airflow DAGs
│   └── india_crop_etl.py        # Daily ETL pipeline
│
├── train/                       # ML training
│   └── train_model.py           # Rule-based + LightGBM models
│
├── api/                         # REST API
│   └── main.py                  # FastAPI endpoints
│
├── ui/                          # Web interface
│   └── index.html               # React/Tailwind UI (20 sensor inputs)
│
└── states.csv/                  # Soil moisture CSV datasets
    ├── sm_Maharashtra_2020.csv
    ├── sm_Gujarat_2020.csv
    ├── sm_Punjab_2020.csv
    ├── sm_rajasthan_2020.csv
    ├── sm_Tamilnadu_2020.csv
    ├── sm_Telangana_2020.csv
    ├── sm_UttarPradesh_2020.csv
    ├── sm_Uttarakhand_2020.csv
    ├── sm_Westbengal_2020.csv
    ├── sm_Andhrapradesh_2020.csv
    └── sm_himachalPradesh_2020.csv
```

## 🚀 Quick Start

### Option 1: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Open UI in browser
# Open ui/index.html in your browser
```

### Option 2: Docker Compose (Full Stack)

```bash
# 1. Start all services (Airflow + Postgres + API)
docker-compose up -d

# 2. Access services
# - API: http://localhost:8000
# - Airflow: http://localhost:8080 (admin/admin)
# - Postgres: localhost:5432
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# API Keys
OPENWEATHERMAP_API_KEY=your_openweather_key
VISUALCROSSING_API_KEY=your_visualcrossing_key

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=india_crop_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Paths
DATA_DIR=./data
MODEL_DIR=./models
```

## 📊 Data Sources

| Source | Type | Format | Access |
|--------|------|--------|--------|
| `states.csv/` | Soil Moisture | CSV | Local folder |
| OpenWeatherMap | Weather | JSON | API (free tier: 1000 calls/day) |
| VisualCrossing | Historical Weather | JSON | API (1000 records/day free) |
| data.gov.in | Crop Statistics | CSV | Public download |
| ICRISAT | Crop Yield | Excel | Open access |

### CSV Dataset Format (states.csv)

| Column | Description | Unit |
|--------|-------------|------|
| Date | Observation date | YYYY/MM/DD |
| State Name | Indian state | String |
| DistrictName | District | String |
| Average Soilmoisture Level (at 15cm) | Mean reading | cm³/cm³ |
| Volume Soilmoisture percentage (at 15cm) | Volumetric % | % |

## 🌾 Supported Crops

| Crop | Season | Soil Moisture Range | Temperature Range |
|------|--------|---------------------|-------------------|
| Rice | Kharif | 30-80% | 20-35°C |
| Wheat | Rabi | 20-50% | 10-25°C |
| Maize | Kharif | 25-60% | 18-32°C |
| Cotton | Kharif | 20-50% | 20-40°C |
| Sugarcane | Perennial | 40-70% | 20-35°C |
| Groundnut | Kharif | 20-45% | 25-35°C |
| Soybean | Kharif | 30-60% | 20-30°C |
| Mustard | Rabi | 15-40% | 10-25°C |
| Chickpea | Rabi | 15-35% | 15-30°C |
| Potato | Rabi | 25-50% | 15-25°C |

## 🖥️ Web UI Features

The web interface supports **manual entry of 20 soil moisture sensor readings**:

- **20 Sensor Input Fields**: Enter soil moisture % from CSV data or manual measurements
- **Bulk Actions**: Set all values, clear all, random fill, load sample data
- **Statistics Panel**: Real-time avg/min/max calculation
- **State Selection**: 11 states with CSV sample data
- **Offline Mode**: Works without API using local rule-based model
- **Sample Data**: Pre-loaded values from actual CSV datasets

### Usage Flow

1. Select a **State** (e.g., Maharashtra)
2. Click **Load Sample** to populate 20 sensors from CSV data
3. Optionally enter **Temperature**, **Rainfall**, **Humidity**
4. Click **Get Crop Recommendations**
5. View top 3 recommended crops with confidence scores

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend` | POST | Get crop recommendations |
| `/status` | GET | API health check |
| `/states` | GET | List supported states |
| `/districts?state=X` | GET | Get districts for state |
| `/crops` | GET | List supported crops |

### Example Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Maharashtra",
    "district": "Pune",
    "soil_moisture_pct": 35.5,
    "temperature_c": 28,
    "rainfall_mm": 100,
    "irrigation_available": true,
    "sensor_readings": [21.4, 15.2, 18.7, ...],
    "num_sensors": 20
  }'
```

### Example Response

```json
{
  "request_id": "abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "location": {"state": "Maharashtra", "district": "Pune"},
  "recommendations": [
    {"crop": "cotton", "confidence": 0.85, "season": "kharif", "notes": "Good conditions"},
    {"crop": "groundnut", "confidence": 0.78, "season": "kharif", "notes": "May need irrigation"},
    {"crop": "soybean", "confidence": 0.72, "season": "kharif", "notes": "Good conditions"}
  ],
  "weather_summary": {"soil_moisture_pct": 35.5, "temperature_c": 28, "season": "kharif"},
  "model_version": "lightgbm_v1"
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest ingest/test_weather_api.py -v

# Test API endpoint
python -c "
import requests
r = requests.get('http://localhost:8000/status')
print(r.json())
"
```

## 📅 ETL Schedule

The Airflow DAG (`india_crop_etl.py`) runs daily with this task flow:

```
ingest_weather_task → ingest_soil_task → ingest_crop_task
                              ↓
                      validate_data_task
                              ↓
                      deduplicate_task
                              ↓
                      load_postgres_task
```

## ✅ Setup Checklist

- [ ] Clone repository
- [ ] Create `.env` file with API keys
- [ ] Install Python 3.11+
- [ ] `pip install -r requirements.txt`
- [ ] Start API: `uvicorn api.main:app --reload`
- [ ] Open `ui/index.html` in browser
- [ ] (Optional) Start Airflow: `docker-compose up -d`
- [ ] (Optional) Train ML model: `python train/train_model.py`

## 📈 Model Performance

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Rule-based | 72% | 0.68 | 0s |
| LightGBM | 85% | 0.82 | 45s |

## 🔒 Security Notes

- API keys should never be committed to git
- Use `.env` files for local development
- Use Docker secrets for production
- Postgres passwords should be changed in production

## 📝 License

MIT License - Built for Indian Agriculture 🇮🇳

## 👥 Contributors

- AI Smart Irrigation Team

---

**Note**: This system uses CSV datasets from the `states.csv/` folder for historical soil moisture data. The web UI allows manual entry of 20 sensor readings to simulate field sensor data, which is then used for crop recommendations.
