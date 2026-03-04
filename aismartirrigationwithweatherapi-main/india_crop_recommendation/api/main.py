"""
FastAPI Application for Crop Recommendation
India Crop Recommendation System

Database: postgresql://postgres:root@localhost:5432/smartirrigationweatherapi

PROMPT 8: REST API with:
- POST /recommend - crop recommendations
- GET /status - health check
- Pydantic models for request/response
- SQLAlchemy DB integration for soil moisture & predictions
"""
import os
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func
from contextlib import asynccontextmanager
import uvicorn

# Database imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from india_crop_recommendation.database import get_db, init_db, engine
from india_crop_recommendation.models import (
    SoilMoistureReading, IrrigationLog, WeatherDaily,
    CropStatistic, ModelPrediction, RefState, RefDistrict,
)
from india_crop_recommendation.pydantic_schemas import (
    SoilMoistureRead, SoilMoistureFilter, SoilMoistureSummary,
    IrrigationLogRead, CSVIngestionResult,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_VERSION = "1.0.0"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Import models (with fallback)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train.train_model import RuleBasedCropRecommender, CropRecommenderML
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    log.warning("ML models not available, using embedded rule-based")

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = API_VERSION
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models_loaded: bool = False


class CropRecommendation(BaseModel):
    crop: str
    confidence: float = Field(ge=0, le=1)
    season: Optional[str] = None
    notes: Optional[str] = None


class RecommendRequest(BaseModel):
    state: str = Field(..., description="Indian state name", examples=["Maharashtra"])
    district: Optional[str] = Field(default=None, description="District name", examples=["Pune"])
    planting_date: Optional[str] = Field(default=None, description="Target planting date (YYYY-MM-DD)")
    soil_moisture_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Soil moisture percentage")
    temperature_c: Optional[float] = Field(default=None, description="Current/expected temperature")
    rainfall_mm: Optional[float] = Field(default=None, ge=0, description="Recent/expected rainfall")
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Humidity percentage")
    budget_inr: Optional[float] = Field(default=None, ge=0, description="Budget in INR")
    land_size_ha: Optional[float] = Field(default=None, ge=0, description="Land size in hectares")
    irrigation_available: bool = Field(default=True, description="Irrigation availability")
    sensor_readings: Optional[List[float]] = Field(default=None, description="20 sensor readings")
    num_sensors: Optional[int] = Field(default=None, description="Number of sensors")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "state": "Maharashtra",
                "district": "Pune",
                "planting_date": "2024-06-15",
                "soil_moisture_pct": 35.0,
                "temperature_c": 28.5,
                "rainfall_mm": 50.0,
                "irrigation_available": True
            }]
        }
    }


class RecommendResponse(BaseModel):
    request_id: str
    timestamp: datetime
    location: Dict[str, str]
    recommendations: List[CropRecommendation]
    weather_summary: Optional[Dict[str, Any]] = None
    model_version: str = "rule_based_v1"
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "request_id": "abc123",
                "timestamp": "2024-01-15T10:30:00",
                "location": {"state": "Maharashtra", "district": "Pune"},
                "recommendations": [
                    {"crop": "cotton", "confidence": 0.85, "season": "kharif", "notes": "Ideal conditions"},
                    {"crop": "soybean", "confidence": 0.75, "season": "kharif", "notes": "Good alternative"},
                    {"crop": "groundnut", "confidence": 0.65, "season": "kharif", "notes": "Consider if irrigation limited"}
                ],
                "model_version": "rule_based_v1"
            }]
        }
    }


class StateInfo(BaseModel):
    name: str
    code: str
    lat: float
    lon: float


class WeatherResponse(BaseModel):
    state: str
    date_str: str
    temp_min_c: float
    temp_max_c: float
    temp_mean_c: float
    precip_mm: float
    humidity_pct: float
    source: str


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables; Shutdown: nothing special."""
    try:
        init_db()
        log.info("PostgreSQL database (smartirrigationweatherapi) initialized.")
    except Exception as exc:
        log.warning("PostgreSQL not available – DB features disabled: %s", exc)
    yield


app = FastAPI(
    title="India Crop Recommendation API",
    description="""
    AI-powered crop recommendation system for Indian agriculture.
    
    ## Features
    - Get personalized crop recommendations based on location, soil, and weather
    - Supports all Indian states
    - Multiple model options (rule-based and ML)
    - PostgreSQL database (smartirrigationweatherapi) for soil moisture, weather, predictions
    
    ## Usage
    1. Call `/recommend` with your location and conditions
    2. Receive top 3 crop recommendations with confidence scores
    3. Use `/db/soil-moisture` to query ingested CSV data from the database
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

STATES = {
    "Andhra Pradesh": {"code": "AP", "lat": 15.9129, "lon": 79.7400},
    "Assam": {"code": "AS", "lat": 26.2006, "lon": 92.9376},
    "Bihar": {"code": "BR", "lat": 25.0961, "lon": 85.3131},
    "Gujarat": {"code": "GJ", "lat": 22.2587, "lon": 71.1924},
    "Haryana": {"code": "HR", "lat": 29.0588, "lon": 76.0856},
    "Karnataka": {"code": "KA", "lat": 15.3173, "lon": 75.7139},
    "Kerala": {"code": "KL", "lat": 10.8505, "lon": 76.2711},
    "Madhya Pradesh": {"code": "MP", "lat": 22.9734, "lon": 78.6569},
    "Maharashtra": {"code": "MH", "lat": 19.7515, "lon": 75.7139},
    "Odisha": {"code": "OR", "lat": 20.9517, "lon": 85.0985},
    "Punjab": {"code": "PB", "lat": 31.1471, "lon": 75.3412},
    "Rajasthan": {"code": "RJ", "lat": 27.0238, "lon": 74.2179},
    "Tamil Nadu": {"code": "TN", "lat": 11.1271, "lon": 78.6569},
    "Telangana": {"code": "TS", "lat": 18.1124, "lon": 79.0193},
    "Uttar Pradesh": {"code": "UP", "lat": 26.8467, "lon": 80.9462},
    "West Bengal": {"code": "WB", "lat": 22.9868, "lon": 87.8550},
}

# Crop requirements for rule-based recommendations
CROP_RULES = {
    "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "precip_min": 100, "season": "kharif"},
    "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "precip_min": 40, "season": "rabi"},
    "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "precip_min": 50, "season": "kharif"},
    "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "precip_min": 60, "season": "kharif"},
    "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "precip_min": 150, "season": "perennial"},
    "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "precip_min": 50, "season": "kharif"},
    "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "precip_min": 60, "season": "kharif"},
    "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "precip_min": 25, "season": "rabi"},
    "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "precip_min": 30, "season": "rabi"},
    "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "precip_min": 40, "season": "rabi"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_season(month: int) -> str:
    """Determine season from month."""
    if 6 <= month <= 10:
        return "kharif"
    elif month >= 10 or month <= 3:
        return "rabi"
    else:
        return "zaid"


def score_crop(
    crop: str,
    soil_moisture: float,
    temp: float,
    precip: float,
    month: int
) -> float:
    """Score crop suitability."""
    if crop not in CROP_RULES:
        return 0.0
    
    rules = CROP_RULES[crop]
    score = 0.0
    
    # Soil moisture (0-30)
    if rules["sm_min"] <= soil_moisture <= rules["sm_max"]:
        mid = (rules["sm_min"] + rules["sm_max"]) / 2
        dist = abs(soil_moisture - mid) / (rules["sm_max"] - rules["sm_min"])
        score += 30 * (1 - dist)
    
    # Temperature (0-30)
    if rules["temp_min"] <= temp <= rules["temp_max"]:
        mid = (rules["temp_min"] + rules["temp_max"]) / 2
        dist = abs(temp - mid) / (rules["temp_max"] - rules["temp_min"])
        score += 30 * (1 - dist)
    
    # Precipitation (0-20)
    if precip >= rules["precip_min"]:
        score += 20
    else:
        score += 20 * (precip / rules["precip_min"])
    
    # Season (0-20)
    if rules["season"] == get_season(month) or rules["season"] == "perennial":
        score += 20
    
    return score


def get_recommendations(
    soil_moisture: float,
    temp: float,
    precip: float,
    month: int,
    n: int = 3
) -> List[CropRecommendation]:
    """Get top N crop recommendations."""
    scores = {
        crop: score_crop(crop, soil_moisture, temp, precip, month)
        for crop in CROP_RULES
    }
    
    total = sum(scores.values())
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    recommendations = []
    for crop, score in sorted_crops:
        confidence = score / 100.0 if total > 0 else 0
        rules = CROP_RULES[crop]
        
        notes = []
        if soil_moisture < rules["sm_min"]:
            notes.append("Consider irrigation")
        if temp > rules["temp_max"]:
            notes.append("High temperature risk")
        if precip < rules["precip_min"]:
            notes.append("May need supplemental irrigation")
        
        recommendations.append(CropRecommendation(
            crop=crop,
            confidence=round(confidence, 3),
            season=rules["season"],
            notes="; ".join(notes) if notes else "Good conditions"
        ))
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """API root - redirects to docs."""
    return {"message": "India Crop Recommendation API", "docs": "/docs"}


@app.get("/status", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    models_loaded = MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib"))
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        models_loaded=models_loaded
    )


@app.get("/states", response_model=List[StateInfo], tags=["Reference"])
async def list_states():
    """List all supported Indian states."""
    return [
        StateInfo(name=name, code=info["code"], lat=info["lat"], lon=info["lon"])
        for name, info in STATES.items()
    ]


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend_crops(request: RecommendRequest):
    """
    Get crop recommendations based on location and conditions.
    
    Accepts state, district, soil moisture, temperature, and rainfall.
    Returns top 3 crop recommendations with confidence scores.
    """
    import uuid
    
    # Validate state
    if request.state not in STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown state: {request.state}. Use /states to list valid states."
        )
    
    # Get date info
    target_date = (
        datetime.strptime(request.planting_date, "%Y-%m-%d").date()
        if request.planting_date
        else date.today()
    )
    month = target_date.month
    
    # Use provided values or defaults
    soil_moisture = request.soil_moisture_pct or 35.0
    temp = request.temperature_c or 25.0
    precip = request.rainfall_mm or 50.0
    
    # Adjust defaults by season
    if get_season(month) == "kharif":
        precip = request.rainfall_mm or 100.0
        temp = request.temperature_c or 28.0
    elif get_season(month) == "rabi":
        precip = request.rainfall_mm or 30.0
        temp = request.temperature_c or 20.0
    
    # Get recommendations
    recommendations = get_recommendations(soil_moisture, temp, precip, month, n=3)
    
    return RecommendResponse(
        request_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow(),
        location={
            "state": request.state,
            "district": request.district or "N/A"
        },
        recommendations=recommendations,
        weather_summary={
            "soil_moisture_pct": soil_moisture,
            "temperature_c": temp,
            "rainfall_mm": precip,
            "season": get_season(month)
        },
        model_version="rule_based_v1"
    )


@app.get("/recommend/quick", response_model=RecommendResponse, tags=["Recommendations"])
async def quick_recommend(
    state: str = Query(..., description="State name"),
    month: int = Query(None, ge=1, le=12, description="Month (1-12)"),
    soil_moisture: float = Query(35.0, ge=0, le=100, description="Soil moisture %")
):
    """Quick recommendation endpoint with minimal parameters."""
    import uuid
    
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    
    month = month or datetime.now().month
    
    # Season-based defaults
    if get_season(month) == "kharif":
        temp, precip = 28.0, 100.0
    else:
        temp, precip = 20.0, 30.0
    
    recommendations = get_recommendations(soil_moisture, temp, precip, month, n=3)
    
    return RecommendResponse(
        request_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow(),
        location={"state": state, "district": "N/A"},
        recommendations=recommendations,
        model_version="rule_based_v1"
    )


@app.get("/weather/{state}", response_model=WeatherResponse, tags=["Weather"])
async def get_weather(state: str):
    """Get current weather for a state (simulated)."""
    import random
    
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    
    # Simulated weather
    month = datetime.now().month
    base_temp = 25 - (STATES[state]["lat"] - 20) * 0.5
    
    if 6 <= month <= 9:  # Monsoon
        temp_mean = base_temp + random.uniform(2, 5)
        precip = random.uniform(50, 200)
        humidity = random.uniform(70, 95)
    else:
        temp_mean = base_temp + random.uniform(-5, 5)
        precip = random.uniform(0, 30)
        humidity = random.uniform(40, 70)
    
    return WeatherResponse(
        state=state,
        date_str=str(date.today()),
        temp_min_c=round(temp_mean - 5, 1),
        temp_max_c=round(temp_mean + 5, 1),
        temp_mean_c=round(temp_mean, 1),
        precip_mm=round(precip, 1),
        humidity_pct=round(humidity, 1),
        source="simulated"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROK CHATBOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

from .grok_chat import get_chatbot, GrokChatBot


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", examples=["What crops should I grow in Maharashtra?"])
    state: Optional[str] = Field(default=None, description="Current state for context")
    district: Optional[str] = Field(default=None, description="Current district")
    soil_moisture_pct: Optional[float] = Field(default=None, description="Current soil moisture")
    temperature_c: Optional[float] = Field(default=None, description="Current temperature")
    rainfall_mm: Optional[float] = Field(default=None, description="Recent rainfall")
    sensor_readings: Optional[List[float]] = Field(default=None, description="Sensor readings array")
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Current month")
    api_key: Optional[str] = Field(default=None, description="Grok API key (optional, overrides env var)")


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_grok(request: ChatRequest):
    """
    Chat with Grok AI for agricultural advice.
    
    Send natural language questions about:
    - Crop recommendations
    - Soil and water management
    - Weather-based farming decisions
    - Regional agricultural practices
    
    Optionally provide context (location, soil moisture, etc.) for more relevant advice.
    You can pass your Grok API key in the request (api_key field) or set it via GROK_API_KEY env var.
    """
    # Get chatbot with optional API key override
    chatbot = get_chatbot(api_key=request.api_key)
    
    # Build context from request
    context = {}
    if request.state:
        context["state"] = request.state
    if request.district:
        context["district"] = request.district
    if request.soil_moisture_pct is not None:
        context["soil_moisture_pct"] = request.soil_moisture_pct
    if request.temperature_c is not None:
        context["temperature_c"] = request.temperature_c
    if request.rainfall_mm is not None:
        context["rainfall_mm"] = request.rainfall_mm
    if request.sensor_readings:
        context["sensor_readings"] = request.sensor_readings
    if request.month is not None:
        context["month"] = request.month
    else:
        context["month"] = datetime.now().month
    
    result = await chatbot.chat_async(request.message, context if context else None)
    
    return ChatResponse(
        success=result["success"],
        response=result.get("response"),
        error=result.get("error")
    )


@app.post("/chat/clear", tags=["Chatbot"])
async def clear_chat_history():
    """Clear the chat conversation history."""
    chatbot = get_chatbot()
    chatbot.clear_history()
    return {"success": True, "message": "Chat history cleared"}


@app.get("/chat/history", tags=["Chatbot"])
async def get_chat_history():
    """Get the current chat conversation history."""
    chatbot = get_chatbot()
    return {"history": chatbot.get_history()}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

from .rag_system import get_rag_system, RAGSystem
from fastapi import UploadFile, File, Form


class RAGUploadResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    filename: str
    num_chunks: int = 0
    message: str
    error: Optional[str] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Search query", examples=["How to irrigate cotton?"])
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    retrieval_method: str = Field(default="hybrid", description="Method: semantic, keyword, or hybrid")
    semantic_weight: float = Field(default=0.5, ge=0, le=1, description="Weight for semantic search")
    keyword_weight: float = Field(default=0.5, ge=0, le=1, description="Weight for keyword search")
    # Metadata filters
    state: Optional[str] = Field(default=None, description="Filter by state")
    crop: Optional[str] = Field(default=None, description="Filter by crop")
    category: Optional[str] = Field(default=None, description="Filter by category")
    date_from: Optional[str] = Field(default=None, description="Filter from date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Filter to date (YYYY-MM-DD)")


class RAGQueryResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    retrieval_method: str
    metadata: Dict[str, Any]


class RAGQueryResponse(BaseModel):
    success: bool
    query: str
    results: List[RAGQueryResult]
    total_results: int
    message: Optional[str] = None


class RAGChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    use_rag: bool = Field(default=True, description="Whether to use RAG context")
    top_k: int = Field(default=5, description="Number of RAG results for context")
    # Context from user
    state: Optional[str] = None
    crop: Optional[str] = None
    # API key
    api_key: Optional[str] = Field(default=None, description="Grok/Groq API key")


class RAGChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    rag_context_used: bool = False
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None


@app.post("/rag/upload", response_model=RAGUploadResponse, tags=["RAG"])
async def upload_document(
    file: UploadFile = File(...),
    state: Optional[str] = Form(default=None),
    crop: Optional[str] = Form(default=None),
    category: Optional[str] = Form(default=None),
    source: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None)
):
    """
    Upload a document to the RAG knowledge base.
    
    Supported formats: PDF, TXT, CSV, JSON, Markdown
    
    The document will be:
    1. Processed and text extracted
    2. Chunked into smaller segments
    3. Embedded for semantic search
    4. Indexed for keyword search
    
    Add metadata (state, crop, category) to enable filtering in queries.
    """
    try:
        rag = get_rag_system()
        
        # Read file content
        content = await file.read()
        
        # Build metadata
        metadata = {}
        if state:
            metadata["state"] = state
        if crop:
            metadata["crop"] = crop
        if category:
            metadata["category"] = category
        if source:
            metadata["source"] = source
        if description:
            metadata["description"] = description
        
        # Add document
        fname = file.filename or "unknown"
        doc = rag.add_document(
            filename=fname,
            content=content,
            metadata=metadata
        )
        
        return RAGUploadResponse(
            success=True,
            document_id=doc.id,
            filename=fname,
            num_chunks=len(doc.chunks),
            message=f"Document uploaded successfully with {len(doc.chunks)} chunks"
        )
        
    except Exception as e:
        log.error(f"RAG upload error: {e}")
        return RAGUploadResponse(
            success=False,
            filename=file.filename or "unknown" if file else "unknown",
            message="Upload failed",
            error=str(e)
        )


@app.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def query_rag(request: RAGQueryRequest):
    """
    Query the RAG knowledge base.
    
    Retrieval methods:
    - **semantic**: Uses embeddings for meaning-based search
    - **keyword**: Uses BM25 for exact keyword matching
    - **hybrid**: Combines both methods (recommended)
    
    Metadata filters can be applied to narrow results by state, crop, category, or date range.
    """
    try:
        rag = get_rag_system()
        
        # Build metadata filters
        filters = {}
        if request.state:
            filters["state"] = request.state
        if request.crop:
            filters["crop"] = request.crop
        if request.category:
            filters["category"] = request.category
        if request.date_from:
            filters["date_from"] = request.date_from
        if request.date_to:
            filters["date_to"] = request.date_to
        
        # Query
        results = rag.query(
            query=request.query,
            top_k=request.top_k,
            retrieval_method=request.retrieval_method,
            metadata_filters=filters if filters else None,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight
        )
        
        # Convert results
        query_results = [
            RAGQueryResult(
                chunk_id=r.chunk.id,
                document_id=r.chunk.document_id,
                content=r.chunk.content,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.chunk.metadata
            )
            for r in results
        ]
        
        return RAGQueryResponse(
            success=True,
            query=request.query,
            results=query_results,
            total_results=len(query_results)
        )
        
    except Exception as e:
        log.error(f"RAG query error: {e}")
        return RAGQueryResponse(
            success=False,
            query=request.query,
            results=[],
            total_results=0,
            message=str(e)
        )


@app.post("/rag/chat", response_model=RAGChatResponse, tags=["RAG"])
async def rag_chat(request: RAGChatRequest):
    """
    Chat with RAG-enhanced responses.
    
    This endpoint:
    1. Retrieves relevant context from the knowledge base
    2. Augments the prompt with this context
    3. Sends to LLM (Grok/Groq) for a grounded response
    
    The response will cite sources from the knowledge base when available.
    """
    try:
        rag = get_rag_system()
        chatbot = get_chatbot(api_key=request.api_key)
        
        rag_context = ""
        sources = []
        
        if request.use_rag:
            # Build filters from context
            filters = {}
            if request.state:
                filters["state"] = request.state
            if request.crop:
                filters["crop"] = request.crop
            
            # Get RAG context
            rag_context = rag.get_context_for_llm(
                query=request.message,
                top_k=request.top_k,
                metadata_filters=filters if filters else None
            )
            
            # Get sources for citation
            results = rag.query(
                query=request.message,
                top_k=request.top_k,
                metadata_filters=filters if filters else None
            )
            sources = [
                {
                    "filename": r.chunk.metadata.get("filename", "unknown"),
                    "chunk_id": r.chunk.id,
                    "score": round(r.score, 3)
                }
                for r in results
            ]
        
        # Build context for LLM
        context = {}
        if request.state:
            context["state"] = request.state
        if request.crop:
            context["crop"] = request.crop
        
        # Add RAG context to the message
        if rag_context:
            augmented_message = f"""Based on the following knowledge base context, please answer the user's question.

KNOWLEDGE BASE CONTEXT:
{rag_context}

USER QUESTION: {request.message}

Please provide a helpful, accurate response based on the context provided. If the context doesn't contain relevant information, you can use your general knowledge but mention that."""
        else:
            augmented_message = request.message
        
        # Get LLM response
        result = await chatbot.chat_async(augmented_message, context if context else None)
        
        return RAGChatResponse(
            success=result["success"],
            response=result.get("response"),
            rag_context_used=bool(rag_context),
            sources=sources,
            error=result.get("error")
        )
        
    except Exception as e:
        log.error(f"RAG chat error: {e}")
        return RAGChatResponse(
            success=False,
            error=str(e)
        )


@app.get("/rag/documents", tags=["RAG"])
async def list_rag_documents():
    """List all documents in the RAG knowledge base."""
    try:
        rag = get_rag_system()
        documents = rag.list_documents()
        return {
            "success": True,
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "documents": [], "total": 0}


@app.delete("/rag/documents/{doc_id}", tags=["RAG"])
async def delete_rag_document(doc_id: str):
    """Delete a document from the RAG knowledge base."""
    try:
        rag = get_rag_system()
        success = rag.delete_document(doc_id)
        
        if success:
            return {"success": True, "message": f"Document {doc_id} deleted"}
        else:
            return {"success": False, "message": f"Document {doc_id} not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/rag/stats", tags=["RAG"])
async def get_rag_stats():
    """Get RAG system statistics."""
    try:
        rag = get_rag_system()
        stats = rag.get_stats()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/db/soil-moisture", response_model=List[SoilMoistureRead], tags=["Database"])
async def get_soil_moisture(
    state: Optional[str] = Query(None, description="Filter by state name"),
    district: Optional[str] = Query(None, description="Filter by district"),
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Query soil moisture readings from the database (loaded from CSV files)."""
    q = db.query(SoilMoistureReading)
    if state:
        q = q.filter(SoilMoistureReading.state.ilike(f"%{state}%"))
    if district:
        q = q.filter(SoilMoistureReading.district.ilike(f"%{district}%"))
    if date_from:
        q = q.filter(SoilMoistureReading.date >= date_from)
    if date_to:
        q = q.filter(SoilMoistureReading.date <= date_to)
    return q.order_by(SoilMoistureReading.date.desc()).offset(offset).limit(limit).all()


@app.get("/db/soil-moisture/summary", response_model=List[SoilMoistureSummary], tags=["Database"])
async def soil_moisture_summary(
    db: Session = Depends(get_db),
):
    """Aggregated soil moisture stats per state."""
    rows = (
        db.query(
            SoilMoistureReading.state,
            func.count().label("record_count"),
            func.avg(SoilMoistureReading.soil_moisture_pct).label("mean_moisture_pct"),
            func.min(SoilMoistureReading.soil_moisture_pct).label("min_moisture_pct"),
            func.max(SoilMoistureReading.soil_moisture_pct).label("max_moisture_pct"),
            func.min(SoilMoistureReading.date).label("date_range_start"),
            func.max(SoilMoistureReading.date).label("date_range_end"),
        )
        .group_by(SoilMoistureReading.state)
        .all()
    )
    return [
        SoilMoistureSummary(
            state=r.state,
            record_count=r.record_count,
            mean_moisture_pct=round(float(r.mean_moisture_pct), 2) if r.mean_moisture_pct else None,
            min_moisture_pct=round(float(r.min_moisture_pct), 2) if r.min_moisture_pct else None,
            max_moisture_pct=round(float(r.max_moisture_pct), 2) if r.max_moisture_pct else None,
            date_range_start=r.date_range_start,
            date_range_end=r.date_range_end,
        )
        for r in rows
    ]


@app.get("/db/soil-moisture/districts/{state}", tags=["Database"])
async def soil_moisture_districts(
    state: str,
    db: Session = Depends(get_db),
):
    """Get soil moisture statistics per district for a given state."""
    rows = (
        db.query(
            SoilMoistureReading.district,
            func.count().label("count"),
            func.avg(SoilMoistureReading.soil_moisture_pct).label("mean_pct"),
            func.min(SoilMoistureReading.soil_moisture_pct).label("min_pct"),
            func.max(SoilMoistureReading.soil_moisture_pct).label("max_pct"),
        )
        .filter(SoilMoistureReading.state.ilike(f"%{state}%"))
        .group_by(SoilMoistureReading.district)
        .all()
    )
    return [
        {
            "district": r.district,
            "count": r.count,
            "mean_moisture_pct": round(float(r.mean_pct), 2) if r.mean_pct else None,
            "min_moisture_pct": round(float(r.min_pct), 2) if r.min_pct else None,
            "max_moisture_pct": round(float(r.max_pct), 2) if r.max_pct else None,
        }
        for r in rows
    ]


@app.get("/db/states", tags=["Database"])
async def list_db_states(db: Session = Depends(get_db)):
    """List all states in the ref_states table."""
    return db.query(RefState).all()


@app.get("/db/districts/{state}", tags=["Database"])
async def list_db_districts(state: str, db: Session = Depends(get_db)):
    """List all districts for a state from the ref_districts table."""
    return (
        db.query(RefDistrict)
        .filter(RefDistrict.state_name.ilike(f"%{state}%"))
        .all()
    )


@app.get("/db/irrigation-logs", response_model=List[IrrigationLogRead], tags=["Database"])
async def get_irrigation_logs(
    limit: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Retrieve recent irrigation decision logs."""
    return (
        db.query(IrrigationLog)
        .order_by(IrrigationLog.created_at.desc())
        .limit(limit)
        .all()
    )


@app.post("/db/load-csv", response_model=CSVIngestionResult, tags=["Database"])
async def load_csv_endpoint():
    """Trigger CSV-to-database ingestion of all state soil moisture CSVs."""
    try:
        from india_crop_recommendation.load_csv_to_db import load_csv_to_db
        result = load_csv_to_db()
        if result:
            return CSVIngestionResult(**result)
        raise HTTPException(500, "Ingestion returned no result")
    except Exception as exc:
        raise HTTPException(500, f"CSV ingestion failed: {exc}")


@app.get("/db/record-counts", tags=["Database"])
async def record_counts(db: Session = Depends(get_db)):
    """Get row counts for all major tables."""
    return {
        "soil_moisture_readings": db.query(func.count(SoilMoistureReading.id)).scalar(),
        "irrigation_logs": db.query(func.count(IrrigationLog.id)).scalar(),
        "weather_daily": db.query(func.count(WeatherDaily.id)).scalar(),
        "crop_statistics": db.query(func.count(CropStatistic.id)).scalar(),
        "model_predictions": db.query(func.count(ModelPrediction.id)).scalar(),
        "ref_states": db.query(func.count(RefState.id)).scalar(),
        "ref_districts": db.query(func.count(RefDistrict.id)).scalar(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the API server."""
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
