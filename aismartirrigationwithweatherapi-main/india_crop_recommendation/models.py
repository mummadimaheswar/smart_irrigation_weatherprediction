"""
SQLAlchemy ORM Models for Smart Irrigation Weather API
Database: postgresql://postgres:root@localhost:5432/smartirrigationweatherapi

Tables:
  - soil_moisture_readings   (CSV data from states.csv/)
  - irrigation_logs          (irrigation request/response log)
  - weather_daily            (daily weather observations)
  - crop_statistics          (yearly crop data)
  - ref_states               (state reference lookup)
  - ref_districts            (district reference lookup)
  - model_predictions        (ML model prediction log)
"""
from datetime import datetime, date, timezone
from sqlalchemy import (
    Column, Integer, BigInteger, Float, String, Date, DateTime,
    Text, Numeric, ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow():
    """Timezone-aware UTC now (avoids deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE TABLES
# ─────────────────────────────────────────────────────────────────────────────

class RefState(Base):
    """Reference table for Indian states."""
    __tablename__ = "ref_states"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    state_code = Column(String(10), unique=True, nullable=False)
    state_name = Column(String(100), unique=True, nullable=False)
    lat        = Column(Numeric(9, 6))
    lon        = Column(Numeric(9, 6))
    created_at = Column(DateTime, default=_utcnow)

    districts = relationship("RefDistrict", back_populates="state", lazy="selectin")

    def __repr__(self):
        return f"<RefState {self.state_name}>"


class RefDistrict(Base):
    """Reference table for districts."""
    __tablename__ = "ref_districts"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    district_code = Column(String(20), unique=True, nullable=False)
    district_name = Column(String(150), nullable=False)
    state_id      = Column(Integer, ForeignKey("ref_states.id"))
    state_name    = Column(String(100))
    lat           = Column(Numeric(9, 6))
    lon           = Column(Numeric(9, 6))
    created_at    = Column(DateTime, default=_utcnow)

    state = relationship("RefState", back_populates="districts", lazy="selectin")

    def __repr__(self):
        return f"<RefDistrict {self.district_name}, {self.state_name}>"


# ─────────────────────────────────────────────────────────────────────────────
# SOIL MOISTURE READINGS (from CSV files)
# ─────────────────────────────────────────────────────────────────────────────

class SoilMoistureReading(Base):
    """
    Soil moisture readings ingested from CSV files in states.csv/.
    Maps directly to the CSV columns:
      Date, State Name, DistrictName,
      Average Soilmoisture Level (at 15cm),
      Average SoilMoisture Volume (at 15cm),
      Aggregate Soilmoisture Percentage (at 15cm),
      Volume Soilmoisture percentage (at 15cm)
    """
    __tablename__ = "soil_moisture_readings"

    id                   = Column(BigInteger, primary_key=True, autoincrement=True)
    date                 = Column(Date, nullable=False, index=True)
    year                 = Column(Integer)
    month                = Column(Integer)
    day                  = Column(Integer)
    state                = Column(String(100), nullable=False, index=True)
    district             = Column(String(150), nullable=False, index=True)
    sm_level_15cm        = Column(Numeric(12, 6), comment="Average Soilmoisture Level at 15cm")
    sm_volume_15cm       = Column(Numeric(12, 6), comment="Average SoilMoisture Volume at 15cm")
    sm_pct_agg_15cm      = Column(Numeric(8, 4), comment="Aggregate Soilmoisture Percentage at 15cm")
    sm_pct_vol_15cm      = Column(Numeric(8, 4), comment="Volume Soilmoisture percentage at 15cm")
    soil_moisture_pct    = Column(Numeric(8, 4), comment="Primary soil moisture %")
    source_file          = Column(String(200))
    created_at           = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("ix_sm_state_date", "state", "date"),
        Index("ix_sm_district_date", "district", "date"),
    )

    def __repr__(self):
        return f"<SoilMoistureReading {self.state}/{self.district} {self.date} sm={self.soil_moisture_pct}>"


# ─────────────────────────────────────────────────────────────────────────────
# IRRIGATION LOGS
# ─────────────────────────────────────────────────────────────────────────────

class IrrigationLog(Base):
    """Stores every irrigation request + system response for analytics."""
    __tablename__ = "irrigation_logs"

    id                = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at        = Column(DateTime, default=_utcnow, nullable=False)

    # Location
    lat               = Column(Float, nullable=False)
    lon               = Column(Float, nullable=False)
    state             = Column(String(100))
    district          = Column(String(150))

    # Sensor input
    soil_moisture     = Column(Float, nullable=False)
    soil_type         = Column(String(50))

    # Crop input
    crop_type         = Column(String(50), nullable=False)
    growth_stage      = Column(String(50))
    days_after_sowing = Column(Integer)

    # Decision output
    decision          = Column(String(30), nullable=False)
    reason            = Column(Text)
    advisory          = Column(Text)
    confidence        = Column(Float)

    # Weather context
    rain_24h_mm       = Column(Float)
    temp_avg_c        = Column(Float)
    et0_mm_day        = Column(Float)

    # Full JSON details
    details           = Column(JSONB)

    def __repr__(self):
        return f"<IrrigationLog id={self.id} crop={self.crop_type} decision={self.decision}>"


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER DAILY
# ─────────────────────────────────────────────────────────────────────────────

class WeatherDaily(Base):
    """Daily weather observations."""
    __tablename__ = "weather_daily"

    id             = Column(BigInteger, primary_key=True, autoincrement=True)
    date           = Column(Date, nullable=False)
    state          = Column(String(100), nullable=False)
    district       = Column(String(150))
    lat            = Column(Numeric(9, 6))
    lon            = Column(Numeric(9, 6))

    temp_min_c     = Column(Numeric(5, 2))
    temp_max_c     = Column(Numeric(5, 2))
    temp_mean_c    = Column(Numeric(5, 2))
    precip_mm      = Column(Numeric(8, 2))
    humidity_pct   = Column(Numeric(5, 2))
    wind_speed_ms  = Column(Numeric(5, 2))
    solar_rad_wm2  = Column(Numeric(8, 2))

    source         = Column(String(50))
    created_at     = Column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("date", "state", "district", name="uq_weather_state_district_date"),
        Index("ix_weather_state_date", "state", "date"),
    )

    def __repr__(self):
        return f"<WeatherDaily {self.state} {self.date}>"


# ─────────────────────────────────────────────────────────────────────────────
# CROP STATISTICS (YEARLY)
# ─────────────────────────────────────────────────────────────────────────────

class CropStatistic(Base):
    """Yearly crop statistics per state/district."""
    __tablename__ = "crop_statistics"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    year             = Column(Integer, nullable=False)
    state            = Column(String(100), nullable=False)
    district         = Column(String(150))
    crop_name        = Column(String(50), nullable=False)
    season           = Column(String(20))

    area_ha          = Column(Numeric(12, 2))
    yield_kg_per_ha  = Column(Numeric(10, 2))
    production_tonnes = Column(Numeric(14, 2))

    source           = Column(String(50))
    created_at       = Column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("year", "state", "district", "crop_name", "season",
                         name="uq_crop_stat"),
        Index("ix_crop_state_year", "state", "year"),
    )

    def __repr__(self):
        return f"<CropStatistic {self.crop_name} {self.state} {self.year}>"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ModelPrediction(Base):
    """ML model prediction log."""
    __tablename__ = "model_predictions"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)
    prediction_date     = Column(DateTime, default=_utcnow)
    state               = Column(String(100))
    district            = Column(String(150))
    target_date         = Column(Date)
    soil_moisture_pct   = Column(Numeric(5, 2))

    recommended_crop_1  = Column(String(50))
    confidence_1        = Column(Numeric(5, 4))
    recommended_crop_2  = Column(String(50))
    confidence_2        = Column(Numeric(5, 4))
    recommended_crop_3  = Column(String(50))
    confidence_3        = Column(Numeric(5, 4))

    model_version       = Column(String(30))
    input_features      = Column(JSONB)
    created_at          = Column(DateTime, default=_utcnow)

    def __repr__(self):
        return f"<ModelPrediction {self.recommended_crop_1} conf={self.confidence_1}>"
