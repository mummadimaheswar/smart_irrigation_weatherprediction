"""
Central Database Module - SQLAlchemy engine, session, and Base
Database: postgresql://postgres:root@localhost:5432/smartirrigationweatherapi
"""
import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:root@localhost:5432/smartirrigationweatherapi",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_session():
    """Context-manager: yields a transactional session (auto commit/rollback)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables defined by Base subclasses."""
    from . import models as _  # noqa – ensure models are imported
    Base.metadata.create_all(bind=engine)
    log.info("All database tables created/verified in smartirrigationweatherapi.")
