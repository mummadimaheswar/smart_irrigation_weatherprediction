"""
CSV to PostgreSQL Ingestion Script
Loads all CSV files from states.csv/ into the soil_moisture_readings table,
and populates ref_states and ref_districts reference tables.

Usage:
    python -m india_crop_recommendation.load_csv_to_db
"""
import sys
import glob
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent dir to path so this can run standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from india_crop_recommendation.database import engine, SessionLocal, init_db
from india_crop_recommendation.models import (
    SoilMoistureReading, RefState, RefDistrict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CSV_FOLDER = Path(__file__).parent / "states.csv"

COLUMN_MAP = {
    "Date": "date",
    "State Name": "state",
    "DistrictName": "district",
    "Average Soilmoisture Level (at 15cm)": "sm_level_15cm",
    "Average SoilMoisture Volume (at 15cm)": "sm_volume_15cm",
    "Aggregate Soilmoisture Percentage (at 15cm)": "sm_pct_agg_15cm",
    "Volume Soilmoisture percentage (at 15cm)": "sm_pct_vol_15cm",
}

STATE_ALIASES = {
    "MAHARASHTRA": "Maharashtra",
    "GUJARAT": "Gujarat",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "TAMILNADU": "Tamil Nadu",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "UTTARPRADESH": "Uttar Pradesh",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "WESTBENGAL": "West Bengal",
    "WEST BENGAL": "West Bengal",
    "ANDHRAPRADESH": "Andhra Pradesh",
    "ANDHRA PRADESH": "Andhra Pradesh",
    "HIMACHALPRADESH": "Himachal Pradesh",
    "HIMACHAL PRADESH": "Himachal Pradesh",
}

STATE_CODES = {
    "Andhra Pradesh": "AP",
    "Gujarat": "GJ",
    "Himachal Pradesh": "HP",
    "Maharashtra": "MH",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Tamil Nadu": "TN",
    "Telangana": "TG",
    "Uttarakhand": "UK",
    "Uttar Pradesh": "UP",
    "West Bengal": "WB",
}

STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Gujarat": (22.2587, 71.1924),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Maharashtra": (19.7515, 75.7139),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Uttarakhand": (30.0668, 79.0193),
    "Uttar Pradesh": (26.8467, 80.9462),
    "West Bengal": (22.9868, 87.8550),
}


def _normalize_state(raw: str) -> str:
    """Normalize state name."""
    return STATE_ALIASES.get(raw.strip().upper(), raw.strip().title())


def _load_single_csv(filepath: Path) -> pd.DataFrame:
    """Load and standardize a single CSV file."""
    df = pd.read_csv(filepath)
    df = df.rename(columns=COLUMN_MAP)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # Normalize state
    if "state" in df.columns:
        df["state"] = df["state"].apply(_normalize_state)

    # Normalize district
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title()
        df["district"] = df["district"].replace("Nan", "Unknown")

    # Derive primary soil moisture %
    if "sm_pct_vol_15cm" in df.columns:
        df["soil_moisture_pct"] = pd.to_numeric(df["sm_pct_vol_15cm"], errors="coerce")
    elif "sm_pct_agg_15cm" in df.columns:
        df["soil_moisture_pct"] = pd.to_numeric(df["sm_pct_agg_15cm"], errors="coerce")
    else:
        df["soil_moisture_pct"] = np.nan

    # Numeric conversions
    for col in ["sm_level_15cm", "sm_volume_15cm", "sm_pct_agg_15cm", "sm_pct_vol_15cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Year/Month/Day
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["source_file"] = filepath.name
    return df


def populate_ref_states(session):
    """Insert reference states from the CSV data."""
    existing = {r.state_name for r in session.query(RefState).all()}
    added = 0
    for state_name, code in STATE_CODES.items():
        if state_name not in existing:
            lat, lon = STATE_COORDS.get(state_name, (None, None))
            session.add(RefState(
                state_code=code,
                state_name=state_name,
                lat=lat,
                lon=lon,
            ))
            added += 1
    session.commit()
    log.info("ref_states: %d new states added (%d total).", added, added + len(existing))


def populate_ref_districts(session, all_districts: dict):
    """Insert reference districts discovered from CSV data.

    all_districts: { (state_name, district_name): count }
    """
    existing = {(r.state_name, r.district_name) for r in session.query(RefDistrict).all()}
    state_lookup = {s.state_name: s.id for s in session.query(RefState).all()}
    added = 0
    for (state_name, district_name), _cnt in all_districts.items():
        if (state_name, district_name) not in existing:
            code = STATE_CODES.get(state_name, "XX")
            session.add(RefDistrict(
                district_code=f"{code}_{district_name[:10].upper().replace(' ', '')}",
                district_name=district_name,
                state_id=state_lookup.get(state_name),
                state_name=state_name,
            ))
            added += 1
    session.commit()
    log.info("ref_districts: %d new districts added.", added)


def load_csv_to_db(batch_size: int = 2000):
    """Main function: load every CSV in states.csv/ into the database."""
    init_db()

    csv_files = sorted(glob.glob(str(CSV_FOLDER / "sm_*.csv")))
    if not csv_files:
        log.error("No CSV files found in %s", CSV_FOLDER)
        return

    log.info("Found %d CSV files in %s", len(csv_files), CSV_FOLDER)

    session = SessionLocal()
    all_districts: dict = {}
    total_inserted = 0
    files_ok = []
    errors = []

    # Populate reference states first
    populate_ref_states(session)

    for fpath in csv_files:
        fpath = Path(fpath)
        try:
            df = _load_single_csv(fpath)
            log.info("Loaded %d rows from %s", len(df), fpath.name)

            # Track districts using vectorized unique pairs
            unique_pairs = df[["state", "district"]].drop_duplicates()
            for state_name, district_name in zip(unique_pairs["state"], unique_pairs["district"]):
                key = (state_name, district_name)
                all_districts[key] = all_districts.get(key, 0) + 1

            # Convert DataFrame to list-of-dicts (much faster than iterrows)
            keep_cols = [
                "date", "year", "month", "day", "state", "district",
                "sm_level_15cm", "sm_volume_15cm", "sm_pct_agg_15cm",
                "sm_pct_vol_15cm", "soil_moisture_pct", "source_file",
            ]
            records_dicts = df[keep_cols].to_dict("records")

            # Build ORM objects in batches
            batch: list = []
            for rec in records_dicts:
                d = rec["date"]
                batch.append(SoilMoistureReading(
                    date=d.date() if hasattr(d, "date") else d,
                    year=int(rec["year"]) if pd.notna(rec.get("year")) else None,
                    month=int(rec["month"]) if pd.notna(rec.get("month")) else None,
                    day=int(rec["day"]) if pd.notna(rec.get("day")) else None,
                    state=rec["state"],
                    district=rec["district"],
                    sm_level_15cm=float(rec["sm_level_15cm"]) if pd.notna(rec.get("sm_level_15cm")) else None,
                    sm_volume_15cm=float(rec["sm_volume_15cm"]) if pd.notna(rec.get("sm_volume_15cm")) else None,
                    sm_pct_agg_15cm=float(rec["sm_pct_agg_15cm"]) if pd.notna(rec.get("sm_pct_agg_15cm")) else None,
                    sm_pct_vol_15cm=float(rec["sm_pct_vol_15cm"]) if pd.notna(rec.get("sm_pct_vol_15cm")) else None,
                    soil_moisture_pct=float(rec["soil_moisture_pct"]) if pd.notna(rec.get("soil_moisture_pct")) else None,
                    source_file=rec.get("source_file"),
                ))

                if len(batch) >= batch_size:
                    session.add_all(batch)
                    session.commit()
                    total_inserted += len(batch)
                    batch = []

            # Flush remaining
            if batch:
                session.add_all(batch)
                session.commit()
                total_inserted += len(batch)

            files_ok.append(fpath.name)

        except Exception as exc:
            log.error("Error processing %s: %s", fpath.name, exc)
            errors.append(f"{fpath.name}: {exc}")
            session.rollback()

    # Populate reference districts
    populate_ref_districts(session, all_districts)

    session.close()

    log.info("=" * 60)
    log.info("CSV INGESTION COMPLETE")
    log.info("  Files processed : %d / %d", len(files_ok), len(csv_files))
    log.info("  Total rows      : %d", total_inserted)
    log.info("  States          : %s", sorted({s for s, d in all_districts}))
    log.info("  Districts       : %d unique", len(all_districts))
    if errors:
        log.warning("  Errors          : %s", errors)
    log.info("=" * 60)

    return {
        "total_files": len(csv_files),
        "total_rows_inserted": total_inserted,
        "files_processed": files_ok,
        "errors": errors,
        "states_loaded": sorted({s for s, d in all_districts}),
        "districts_loaded": sorted({d for s, d in all_districts}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = load_csv_to_db()
    if result:
        print(f"\nDone! Inserted {result['total_rows_inserted']} rows from "
              f"{len(result['files_processed'])} CSV files into "
              f"smartirrigationweatherapi database.")
