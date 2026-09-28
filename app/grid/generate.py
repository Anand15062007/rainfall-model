"""Generate approximately square metric grids in a local azimuthal projection."""
import math
import pandas as pd


def generate_grid(city_name: str, bbox: tuple[float, float, float, float], grid_size_km: float = 1.0) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    if grid_size_km <= 0: raise ValueError("grid_size_km must be positive")
    lon0, lat0 = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    # Spherical azimuthal-equidistant projection (WGS84 mean Earth radius).
    # This preserves radial distances from the city center and yields metric square cells.
    radius=6371008.8; phi0=math.radians(lat0); lam0=math.radians(lon0)
    def forward(lon,lat):
        phi=math.radians(lat); lam=math.radians(lon); dl=lam-lam0
        cosc=max(-1.,min(1.,math.sin(phi0)*math.sin(phi)+math.cos(phi0)*math.cos(phi)*math.cos(dl)))
        c=math.acos(cosc); k=1. if c==0 else c/math.sin(c)
        return radius*k*math.cos(phi)*math.sin(dl), radius*k*(math.cos(phi0)*math.sin(phi)-math.sin(phi0)*math.cos(phi)*math.cos(dl))
    def inverse(x,y):
        rho=math.hypot(x,y)
        if rho==0: return lon0,lat0
        c=rho/radius; phi=math.asin(math.cos(c)*math.sin(phi0)+y*math.sin(c)*math.cos(phi0)/rho)
        lam=lam0+math.atan2(x*math.sin(c),rho*math.cos(phi0)*math.cos(c)-y*math.sin(phi0)*math.sin(c))
        return math.degrees(lam),math.degrees(phi)
    corners = [forward(x, y) for x in (min_lon,max_lon) for y in (min_lat,max_lat)]
    xs, ys = zip(*corners); step = grid_size_km * 1000; step_i=round(step)
    rows=[]; i=0
    for y in range(math.floor(min(ys)/step)*step_i, math.ceil(max(ys)/step)*step_i, step_i):
        for x in range(math.floor(min(xs)/step)*step_i, math.ceil(max(xs)/step)*step_i, step_i):
            cx,cy=x+step/2,y+step/2
            clon,clat=inverse(cx,cy)
            vertices=[inverse(x,y),inverse(x+step,y),inverse(x+step,y+step),inverse(x,y+step)]
            lons=[p[0] for p in vertices]; lats=[p[1] for p in vertices]
            if max(lons)<min_lon or min(lons)>max_lon or max(lats)<min_lat or min(lats)>max_lat: continue
            rows.append(dict(grid_id=f"G{i:05d}",city=city_name,latitude_center=clat,longitude_center=clon,
                min_lat=min(lats),max_lat=max(lats),min_lon=min(lons),max_lon=max(lons),grid_size_km=grid_size_km))
            i+=1
    return pd.DataFrame(rows)

if __name__ == "__main__":
    from app.config.settings import load_config
    c=load_config(); g=generate_grid(c['city']['name'],c['city']['bbox'],c['city']['grid_size_km'])
    from pathlib import Path
    Path('data/processed').mkdir(exist_ok=True,parents=True); g.to_csv('data/processed/grids.csv',index=False)
    print(f"Wrote {len(g)} grids to data/processed/grids.csv")
