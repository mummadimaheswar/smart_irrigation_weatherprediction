"""
Pydantic Schemas for API request/response validation
Used by FastAPI endpoints for serialization/deserialization.
"""
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# SOIL MOISTURE
# ─────────────────────────────────────────────────────────────────────────────

class SoilMoistureBase(BaseModel):
    date: date
    state: str
    district: str
    soil_moisture_pct: Optional[float] = None
    sm_level_15cm: Optional[float] = None
    sm_volume_15cm: Optional[float] = None
    sm_pct_agg_15cm: Optional[float] = None
    sm_pct_vol_15cm: Optional[float] = None


class SoilMoistureCreate(SoilMoistureBase):
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    source_file: Optional[str] = None


class SoilMoistureRead(SoilMoistureBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    source_file: Optional[str] = None
    created_at: Optional[datetime] = None


class SoilMoistureFilter(BaseModel):
    """Query filter for soil moisture data."""
    state: Optional[str] = None
    district: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    min_moisture: Optional[float] = None
    max_moisture: Optional[float] = None


class SoilMoistureSummary(BaseModel):
    state: str
    district: Optional[str] = None
    record_count: int
    mean_moisture_pct: Optional[float] = None
    min_moisture_pct: Optional[float] = None
    max_moisture_pct: Optional[float] = None
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None


# ─────────────────────────────────────────────────────────────────────────────
# IRRIGATION
# ─────────────────────────────────────────────────────────────────────────────

class LocationInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    state: Optional[str] = None
    district: Optional[str] = None


class SensorInput(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=1, description="Soil moisture (0-1)")
    soil_type: str = Field("loam", description="Soil type")


class CropInput(BaseModel):
    crop_type: str = Field("wheat", description="Crop type")
    growth_stage: Optional[str] = Field(None, description="Growth stage")
    days_after_sowing: Optional[int] = Field(None, ge=0)


class IrrigationRequest(BaseModel):
    location: LocationInput
    sensor: SensorInput
    crop: CropInput


class IrrigationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    decision: str
    reason: str
    advisory: str
    confidence: float
    details: Dict[str, Any]
    timestamp: str


class IrrigationLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    lat: float
    lon: float
    state: Optional[str] = None
    district: Optional[str] = None
    soil_moisture: float
    soil_type: Optional[str] = None
    crop_type: str
    growth_stage: Optional[str] = None
    days_after_sowing: Optional[int] = None
    decision: str
    reason: Optional[str] = None
    advisory: Optional[str] = None
    confidence: Optional[float] = None
    rain_24h_mm: Optional[float] = None
    temp_avg_c: Optional[float] = None
    et0_mm_day: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────────────────────────────────────

class WeatherDailyBase(BaseModel):
    date: date
    state: str
    district: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None
    temp_mean_c: Optional[float] = None
    precip_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    solar_rad_wm2: Optional[float] = None
    source: Optional[str] = None


class WeatherDailyCreate(WeatherDailyBase):
    pass


class WeatherDailyRead(WeatherDailyBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: Optional[datetime] = None


class WeatherResponse(BaseModel):
    location: Dict[str, Any]
    forecast_24h: Dict[str, Any]
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# CROP STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class CropStatisticBase(BaseModel):
    year: int
    state: str
    district: Optional[str] = None
    crop_name: str
    season: Optional[str] = None
    area_ha: Optional[float] = None
    yield_kg_per_ha: Optional[float] = None
    production_tonnes: Optional[float] = None
    source: Optional[str] = None


class CropStatisticCreate(CropStatisticBase):
    pass


class CropStatisticRead(CropStatisticBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class StateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    state_code: str
    state_name: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class DistrictRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    district_code: str
    district_name: str
    state_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

class PredictionCreate(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    target_date: Optional[date] = None
    soil_moisture_pct: Optional[float] = None
    recommended_crop_1: Optional[str] = None
    confidence_1: Optional[float] = None
    recommended_crop_2: Optional[str] = None
    confidence_2: Optional[float] = None
    recommended_crop_3: Optional[str] = None
    confidence_3: Optional[float] = None
    model_version: Optional[str] = None
    input_features: Optional[Dict[str, Any]] = None


class PredictionRead(PredictionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_date: Optional[datetime] = None
    created_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: str


class PaginatedResponse(BaseModel):
    total: int
    page: int
    per_page: int
    items: List[Any]


class CSVIngestionResult(BaseModel):
    """Result of CSV-to-database ingestion."""
    total_files: int
    total_rows_inserted: int
    files_processed: List[str]
    errors: List[str]
    states_loaded: List[str]
    districts_loaded: List[str]
