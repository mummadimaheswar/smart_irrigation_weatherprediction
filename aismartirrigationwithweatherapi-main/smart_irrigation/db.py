"""PostgreSQL storage for irrigation requests & responses.

Database: postgresql://postgres:root@localhost:5432/smartirrigationweatherapi
"""
import json
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, Float, String, DateTime, Text,
    MetaData, Table, inspect,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

log = logging.getLogger(__name__)

# ── Database URL (shared across the whole project) ──────────────────────────
DATABASE_URL = "postgresql://postgres:root@localhost:5432/smartirrigationweatherapi"

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
# ORM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class IrrigationLog(Base):
    """Stores every user request + system response for analytics."""
    __tablename__ = "irrigation_logs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    created_at    = Column(DateTime, default=_utcnow, nullable=False)

    # ── Location (user input) ───────────────────────────────────────────────
    lat           = Column(Float, nullable=False)
    lon           = Column(Float, nullable=False)
    state         = Column(String(100))
    district      = Column(String(100))

    # ── Sensor (user input) ─────────────────────────────────────────────────
    soil_moisture = Column(Float, nullable=False)
    soil_type     = Column(String(50))

    # ── Crop (user input) ───────────────────────────────────────────────────
    crop_type         = Column(String(50), nullable=False)
    growth_stage      = Column(String(50))
    days_after_sowing = Column(Integer)

    # ── Response ────────────────────────────────────────────────────────────
    decision      = Column(String(30), nullable=False)
    reason        = Column(Text)
    advisory      = Column(Text)
    confidence    = Column(Float)

    # ── Weather context at decision time ────────────────────────────────────
    rain_24h_mm   = Column(Float)
    temp_avg_c    = Column(Float)
    et0_mm_day    = Column(Float)

    # ── Full details blob for flexible analysis ─────────────────────────────
    details       = Column(JSONB)

    def __repr__(self):
        return (
            f"<IrrigationLog id={self.id} crop={self.crop_type} "
            f"decision={self.decision} @ {self.created_at}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE / SESSION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session():
    """Yield a transactional DB session that auto-commits / rolls back."""
    Session = _get_session_factory()
    session = Session()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE CREATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Call once at startup."""
    engine = _get_engine()
    Base.metadata.create_all(engine)
    log.info("PostgreSQL tables ensured (irrigation_logs).")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log_irrigation_request(
    *,
    lat: float,
    lon: float,
    state: str | None,
    district: str | None,
    soil_moisture: float,
    soil_type: str,
    crop_type: str,
    growth_stage: str | None,
    days_after_sowing: int | None,
    decision: str,
    reason: str,
    advisory: str,
    confidence: float,
    rain_24h_mm: float | None = None,
    temp_avg_c: float | None = None,
    et0_mm_day: float | None = None,
    details: dict | None = None,
) -> int | None:
    """Insert one request+response row. Returns the new row id, or None on error."""
    try:
        with get_session() as session:
            row = IrrigationLog(
                lat=lat,
                lon=lon,
                state=state,
                district=district,
                soil_moisture=soil_moisture,
                soil_type=soil_type,
                crop_type=crop_type,
                growth_stage=growth_stage,
                days_after_sowing=days_after_sowing,
                decision=decision,
                reason=reason,
                advisory=advisory,
                confidence=confidence,
                rain_24h_mm=rain_24h_mm,
                temp_avg_c=temp_avg_c,
                et0_mm_day=et0_mm_day,
                details=details,
            )
            session.add(row)
            session.flush()          # populate row.id before commit
            row_id = int(row.id)     # type: ignore[arg-type]
        log.info("Logged irrigation request id=%s", row_id)
        return row_id
    except SQLAlchemyError as exc:
        log.error("Failed to log irrigation request: %s", exc)
        return None
