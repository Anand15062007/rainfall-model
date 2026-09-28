"""Extract CHIRPS v3 daily-final satellite rainfall for the configured bbox.

The NetCDF is read remotely by HTTP byte ranges; only the CHIRPS chunks covering
the requested area are transferred. Outputs one area-mean rainfall value per day.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from app.config.settings import load_config

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/netcdf/byMonth"
CHIRPS_EPOCH = pd.Timestamp("1980-01-01", tz="UTC")


def _monthly_area_mean(year: int, month: int, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    try:
        import fsspec
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install CHIRPS reader dependencies: pip install -e '.[chirps]'") from exc

    west, south, east, north = bbox
    if west > east:
        raise ValueError("A bbox crossing the dateline is not supported; split it into two bboxes")
    filename = f"chirps-v3.0.{year}.{month:02d}.days_p05.nc"
    url = f"{BASE_URL}/{filename}"
    try:
        # HDF5 performs non-sequential seeks into NetCDF; blockcache avoids
        # re-downloading large spans between those reads.
        remote = fsspec.open(url, mode="rb", block_size=1 << 20,
                             cache_type="blockcache").open()
        ds = h5py.File(remote, "r")
    except Exception as exc:
        # Final CHIRPS files are released after the month ends; absent months are skipped.
        if "404" in str(exc) or "FileNotFound" in type(exc).__name__:
            return pd.DataFrame(columns=["date", "rainfall_mm", "chirps_cells"])
        raise RuntimeError(f"Could not open CHIRPS file {filename}: {type(exc).__name__}") from exc

    try:
        lat = ds["latitude"][:]
        lon = ds["longitude"][:]
        times = ds["time"][:]
        precip = ds["precip"]
        # CHIRPS cells are 0.05 degrees wide. Select cells whose footprints
        # intersect the requested bbox, so small city bboxes still select cells.
        dlat = float(np.median(np.diff(lat)))
        dlon = float(np.median(np.diff(lon)))
        lat_ix = np.flatnonzero((lat + dlat / 2 >= south) & (lat - dlat / 2 <= north))
        lon_ix = np.flatnonzero((lon + dlon / 2 >= west) & (lon - dlon / 2 <= east))
        if not len(lat_ix) or not len(lon_ix):
            raise ValueError("bbox does not intersect CHIRPS v3 domain (60S–60N)")
        y0, y1 = int(lat_ix.min()), int(lat_ix.max()) + 1
        x0, x1 = int(lon_ix.min()), int(lon_ix.max()) + 1
        # Use a bounding block for efficient HDF5 chunk reads, then mask cells
        # whose footprints do not actually intersect the bbox.
        sub = precip[:, y0:y1, x0:x1].astype("float32", copy=False)
        sub_lat, sub_lon = lat[y0:y1], lon[x0:x1]
        cell_mask = ((sub_lat[:, None] + dlat / 2 >= south)
                     & (sub_lat[:, None] - dlat / 2 <= north)
                     & (sub_lon[None, :] + dlon / 2 >= west)
                     & (sub_lon[None, :] - dlon / 2 <= east))
        sub[sub == -9999] = np.nan
        sub[:, ~cell_mask] = np.nan
        values = np.nanmean(sub, axis=(1, 2))
        dates = CHIRPS_EPOCH + pd.to_timedelta(times, unit="D")
        return pd.DataFrame({"date": dates.normalize(), "rainfall_mm": values,
                             "chirps_cells": int(cell_mask.sum())})
    finally:
        ds.close()
        remote.close()


def download_chirps(bbox, start_year: int, end_year: int, output: str | Path) -> pd.DataFrame:
    if start_year < 1998 or end_year < start_year:
        raise ValueError("CHIRPS v3 daily satellite product starts in 1998; provide a valid year range")
    rows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            try:
                frame = _monthly_area_mean(year, month, tuple(bbox))
            except Exception as exc:
                raise RuntimeError(f"CHIRPS download failed for {year}-{month:02d}: {exc}") from exc
            if not frame.empty:
                rows.append(frame)
                print(f"Read CHIRPS {year}-{month:02d}: {len(frame)} days, {frame.chirps_cells.iloc[0]} native cells")
    if not rows:
        raise RuntimeError("No CHIRPS files were available for the selected period and bbox")
    result = pd.concat(rows, ignore_index=True).drop_duplicates("date").sort_values("date")
    result = result[(result.date.dt.year >= start_year) & (result.date.dt.year <= end_year)]
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} daily area-mean rainfall records to {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--output", default="data/raw/chirps_v3_daily.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    start = args.start_year or cfg["data"].get("live_start_year", 2015)
    end = args.end_year or cfg["data"].get("live_end_year", 2025)
    download_chirps(cfg["city"]["bbox"], start, end, args.output)


if __name__ == "__main__":
    main()
