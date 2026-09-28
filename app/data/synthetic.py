"""Correlated, explicitly synthetic hourly weather observations."""
import numpy as np
import pandas as pd

OPTIONAL = ["solar_radiation","visibility","soil_moisture","surface_temperature","CAPE","precipitable_water","vertical_velocity","upper_air_temperature","upper_air_humidity"]

def generate_synthetic(grids: pd.DataFrame, start="2022-01-01T00:00:00Z", periods=1000, frequency="1h", seed=42) -> pd.DataFrame:
    if grids.empty: raise ValueError("at least one grid is required")
    rng=np.random.default_rng(seed); times=pd.date_range(start,periods=periods,freq=frequency)
    out=[]
    for gi,g in grids.reset_index(drop=True).iterrows():
        phase=gi*.37; hour=times.hour.to_numpy(); doy=times.dayofyear.to_numpy()
        temp=17+9*np.sin(2*np.pi*(doy-100)/365)+4*np.sin(2*np.pi*(hour-8)/24)+rng.normal(0,1.5,periods)
        rh=np.clip(67-0.8*(temp-17)+12*np.sin(2*np.pi*hour/24+phase)+rng.normal(0,8,periods),20,100)
        pressure=1013+5*np.sin(2*np.pi*doy/365)+rng.normal(0,2,periods)
        cloud=np.clip((rh-35)*1.25+rng.normal(0,18,periods),0,100)
        wind=np.clip(rng.gamma(2,2,periods),0,25); direction=rng.uniform(0,360,periods)
        wet_prob=np.clip(.04+.004*cloud+.002*(rh-60),.02,.8)
        wet=rng.random(periods)<wet_prob
        amount=np.where(wet,rng.gamma(1.5,np.maximum(.25,cloud/65),periods),0)
        amount=np.minimum(amount,45)
        previous=np.r_[0,amount[:-1]]
        frame=pd.DataFrame({"timestamp":times,"grid_id":g.grid_id,"latitude":g.latitude_center,"longitude":g.longitude_center,
            "temperature":temp,"relative_humidity":rh,"pressure":pressure,"wind_speed":wind,"wind_direction":direction,
            "cloud_cover":cloud,"dew_point":temp-(100-rh)/5,"rainfall_1h":amount,"rainfall_3h":pd.Series(amount).rolling(3,min_periods=1).sum().to_numpy(),
            "rainfall_6h":pd.Series(amount).rolling(6,min_periods=1).sum().to_numpy(),"rainfall_12h":pd.Series(amount).rolling(12,min_periods=1).sum().to_numpy(),
            "rainfall_24h":pd.Series(amount).rolling(24,min_periods=1).sum().to_numpy(),"solar_radiation":np.maximum(0,600*np.sin(np.pi*(hour-6)/12)),
            "visibility":np.clip(15-cloud*.12+rng.normal(0,1,periods),.2,20),"soil_moisture":np.clip(.2+np.convolve(amount,np.ones(24)/80,'same'),0,1),
            "surface_temperature":temp+rng.normal(1,1,periods),"CAPE":np.maximum(0,300+cloud*15+rng.normal(0,250,periods)),
            "precipitable_water":np.maximum(0,20+rh*.25+rng.normal(0,3,periods)),"vertical_velocity":rng.normal(-.05,.2,periods),
            "upper_air_temperature":temp-18+rng.normal(0,1,periods),"upper_air_humidity":np.clip(rh-15+rng.normal(0,8,periods),0,100)})
        out.append(frame)
    df=pd.concat(out,ignore_index=True)
    # Realistic sensor gaps, retained as missing data for preprocessing demonstrations.
    gap=rng.random(len(df))<.01; df.loc[gap,"pressure"]=np.nan
    df["rain_event"]=(df["rainfall_1h"]>=.1).astype("int8")
    df["rainfall_amount_mm"]=df["rainfall_1h"]
    return df

if __name__ == "__main__":
    from app.config.settings import load_config
    from app.grid.generate import generate_grid
    from pathlib import Path
    c=load_config(); g=generate_grid(c['city']['name'],c['city']['bbox'],c['city']['grid_size_km'])
    d=generate_synthetic(g,c['data']['start'],c['data']['periods'],c['data']['frequency'],c['random_seed'])
    Path('data/synthetic').mkdir(exist_ok=True,parents=True); d.to_csv('data/synthetic/weather.csv',index=False)
    print(f"Wrote {len(d)} synthetic observations")
