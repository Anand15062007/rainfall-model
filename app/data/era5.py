"""ERA5 hourly single-level NetCDF normalization to the internal observation schema."""
from pathlib import Path
import numpy as np
import pandas as pd

VARIABLE_ALIASES={
    "temperature":["t2m","2m_temperature"],
    "dew_point":["d2m","2m_dewpoint_temperature"],
    "pressure":["msl","mean_sea_level_pressure"],
    "wind_u_10m":["u10","10m_u_component_of_wind"],
    "wind_v_10m":["v10","10m_v_component_of_wind"],
    "cloud_cover":["tcc","total_cloud_cover"],
    "total_precipitation":["tp","total_precipitation"],
    "total_column_water_vapour":["tcwv","total_column_water_vapour"],
    "CAPE":["cape","convective_available_potential_energy"],
    "CIN":["cin","convective_inhibition"],
    "boundary_layer_height":["blh","boundary_layer_height"],
    "geopotential":["z","geopotential"],
}

def _first_present(data_vars, aliases):
    return next((name for name in aliases if name in data_vars),None)

def netcdf_to_observations(path,bbox=None,threshold_mm=.1):
    """Convert one CDS NetCDF file. Rainfall is ERA5 hourly tp (m) converted to mm."""
    try: import xarray as xr
    except ImportError as e: raise ImportError("Install ERA5 dependencies with pip install -r requirements.txt") from e
    ds=xr.open_dataset(path)
    try:
        if "expver" in ds.dims: ds=ds.isel(expver=0,drop=True)
        time_name=next((x for x in ("valid_time","time") if x in ds.coords),None)
        lat_name=next((x for x in ("latitude","lat") if x in ds.coords),None)
        lon_name=next((x for x in ("longitude","lon") if x in ds.coords),None)
        if not all((time_name,lat_name,lon_name)): raise ValueError("NetCDF requires time/valid_time, latitude and longitude coordinates")
        names={out:_first_present(ds.data_vars,aliases) for out,aliases in VARIABLE_ALIASES.items()}
        if names["total_precipitation"] is None: raise ValueError("ERA5 file must include Total precipitation (tp)")
        selected=[x for x in names.values() if x]
        raw=ds[selected].to_dataframe().reset_index()
        raw=raw.rename(columns={time_name:"timestamp",lat_name:"latitude",lon_name:"longitude"})
        if bbox is not None:
            west,south,east,north=bbox
            raw=raw[(raw.longitude>=west)&(raw.longitude<=east)&(raw.latitude>=south)&(raw.latitude<=north)]
        if raw.empty: raise ValueError("No ERA5 grid cells fall inside the configured bounding box")
        out=pd.DataFrame({"timestamp":pd.to_datetime(raw.timestamp,utc=True),"latitude":raw.latitude.astype(float),"longitude":raw.longitude.astype(float)})
        lat_key=out.latitude.map(lambda x:f"{x:.4f}"); lon_key=out.longitude.map(lambda x:f"{x:.4f}")
        out["grid_id"]="ERA5_"+lat_key+"_"+lon_key
        for canonical,var in names.items():
            if var is None: continue
            values=pd.to_numeric(raw[var],errors="coerce")
            if canonical in ("temperature","dew_point"): values=values-273.15
            elif canonical=="pressure": values=values/100.0
            elif canonical=="total_precipitation":
                units=str(ds[var].attrs.get("units","m")).lower()
                if units.startswith("m"): values=values*1000.0
                elif "mm" in units: pass
                else: raise ValueError(f"Unrecognized ERA5 precipitation units {units!r}; expected metres or mm")
                out["rainfall_1h"]=values.clip(lower=0)
            elif canonical=="cloud_cover":
                if values.dropna().max()<=1.01: values=values*100.0
                out[canonical]=values.clip(0,100); continue
            elif canonical=="geopotential":
                out["orography_m"]=values/9.80665; continue
            out[canonical]=values
        # Relative humidity from 2m temperature and dewpoint via Magnus saturation vapour pressure.
        if "temperature" in out and "dew_point" in out:
            es=6.1094*np.exp(17.625*out.temperature/(243.04+out.temperature))
            e=6.1094*np.exp(17.625*out.dew_point/(243.04+out.dew_point))
            out["relative_humidity"]=(100*e/es).clip(0,100)
        if {"wind_u_10m","wind_v_10m"}.issubset(out.columns):
            u=out.wind_u_10m; v=out.wind_v_10m
            out["wind_speed"]=(u*u+v*v)**.5
            out["wind_direction"]=(np.degrees(np.arctan2(-u,-v))+360)%360
        out["rain_event"]=(out.rainfall_1h>=threshold_mm).astype("int8")
        out["rainfall_amount_mm"]=out.rainfall_1h
        return out.sort_values(["grid_id","timestamp"]).reset_index(drop=True)
    finally:
        ds.close()

def grid_catalog_from_observations(frame):
    """Return ERA5 cells at their native coordinate spacing; never fabricate 1-km cells."""
    points=frame[["grid_id","latitude","longitude"]].drop_duplicates("grid_id").copy()
    if points.empty: raise ValueError("no ERA5 grid cells found")
    lat_step=float(np.median(np.diff(np.sort(points.latitude.unique())))) if points.latitude.nunique()>1 else .25
    lon_step=float(np.median(np.diff(np.sort(points.longitude.unique())))) if points.longitude.nunique()>1 else .25
    rows=[]
    for r in points.itertuples(index=False):
        lat_km=abs(lat_step)*111.32; lon_km=abs(lon_step)*111.32*np.cos(np.deg2rad(r.latitude)); size=(lat_km*lon_km)**.5
        rows.append({"grid_id":r.grid_id,"city":"ERA5 native grid","latitude_center":r.latitude,"longitude_center":r.longitude,
            "min_lat":r.latitude-lat_step/2,"max_lat":r.latitude+lat_step/2,"min_lon":r.longitude-lon_step/2,"max_lon":r.longitude+lon_step/2,"grid_size_km":size})
    return pd.DataFrame(rows)
