"""Convert downloaded ERA5 monthly NetCDF files into standardized hourly CSV."""
import argparse
from pathlib import Path
import pandas as pd
from app.config.settings import load_config
from app.data.era5 import netcdf_to_observations,grid_catalog_from_observations

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",default="data/raw/era5"); p.add_argument("--output",default="data/processed/era5_observations.csv"); p.add_argument("--config",default="configs/default.yaml"); a=p.parse_args()
    c=load_config(a.config); west,south,east,north=c["city"]["bbox"]
    paths=sorted(Path(a.input_dir).glob("*.nc"))
    if not paths: raise FileNotFoundError(f"No .nc files found in {a.input_dir}")
    frames=[netcdf_to_observations(path,(west,south,east,north),c["data"]["rainfall_threshold_mm"]) for path in paths]
    df=pd.concat(frames,ignore_index=True).drop_duplicates(["timestamp","grid_id"]).sort_values(["grid_id","timestamp"])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False)
    grid_catalog_from_observations(df).to_csv("data/processed/grids.csv",index=False)
    print(f"Wrote {len(df)} rows from {len(paths)} ERA5 files to {a.output}; {df.grid_id.nunique()} native ERA5 cells")
if __name__=="__main__": main()
