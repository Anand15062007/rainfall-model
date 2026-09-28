"""Download compact monthly ERA5 single-level subsets from CDS after .env is configured."""
import argparse, calendar, os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_exponential
from app.config.settings import load_config

DATASET="reanalysis-era5-single-levels"
# Parsimonious meteorological predictors plus the precipitation target and topography.
VARIABLES=["2m_temperature","2m_dewpoint_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
 "mean_sea_level_pressure","total_cloud_cover","total_precipitation","total_column_water_vapour",
 "convective_available_potential_energy","convective_inhibition","boundary_layer_height","geopotential"]
TIMES=[f"{h:02d}:00" for h in range(24)]

@retry(stop=stop_after_attempt(5),wait=wait_exponential(multiplier=2,min=2,max=60),reraise=True)
def _retrieve(client,request,target):
    client.retrieve(DATASET,request,str(target))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--start-year",type=int); p.add_argument("--end-year",type=int); p.add_argument("--output-dir",default="data/raw/era5"); a=p.parse_args()
    load_dotenv(); token=os.getenv("CDS_API_KEY")
    if not token: raise SystemExit("CDS_API_KEY is missing. Add it to the project .env after accepting the CDS dataset licence.")
    try: import cdsapi
    except ImportError as e: raise SystemExit("Install ERA5 tools with: pip install -r requirements.txt") from e
    cfg=load_config(a.config); start=a.start_year or cfg["data"].get("era5_start_year",2015); end=a.end_year or cfg["data"].get("era5_end_year",2025)
    if end<start: raise SystemExit("end year must be >= start year")
    west,south,east,north=cfg["city"]["bbox"]; area=[north,west,south,east]
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    client=cdsapi.Client(url="https://cds.climate.copernicus.eu/api",key=token)
    for year in range(start,end+1):
        for month in range(1,13):
            target=outdir/f"era5_{year}_{month:02d}.nc"
            if target.exists() and target.stat().st_size>0:
                print(f"Skip existing {target}"); continue
            days=[f"{d:02d}" for d in range(1,calendar.monthrange(year,month)[1]+1)]
            request={"product_type":["reanalysis"],"variable":VARIABLES,"year":[str(year)],"month":[f"{month:02d}"],"day":days,"time":TIMES,"area":area,"data_format":"netcdf","download_format":"unarchived"}
            print(f"Requesting {year}-{month:02d} for bbox N/W/S/E={area}")
            _retrieve(client,request,target)
if __name__=="__main__": main()
