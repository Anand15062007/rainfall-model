"""Reusable, file-based evaluation figures."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def rainfall_series(df,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); s=df.groupby("timestamp").rainfall_1h.mean(); ax=s.plot(figsize=(11,4),title="Mean synthetic rainfall (mm/hour)"); ax.set_ylabel("mm/hour"); plt.tight_layout(); plt.savefig(p); plt.close()

def comparison_plot(metrics,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); d=pd.DataFrame(metrics).T.apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all"); ax=d.plot(kind="bar",figsize=(10,5),title="Model metrics (different scales; compare columns individually)"); ax.set_ylim(bottom=0); plt.tight_layout(); plt.savefig(p); plt.close()

def grid_map(grids,predictions,path="data/processed/grid_map.html"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    pred=predictions.set_index("grid_id") if predictions is not None and not predictions.empty else None
    try:
        import folium
        m=folium.Map(location=[grids.latitude_center.mean(),grids.longitude_center.mean()],zoom_start=11,tiles="CartoDB positron")
        for _,g in grids.iterrows():
            d=pred.loc[g.grid_id] if pred is not None and g.grid_id in pred.index else None
            rain=float(d.get("predicted_rainfall_mm",0)) if d is not None else 0
            prob=float(d.get("rain_probability",0)) if d is not None else 0
            color="#d73027" if rain>=10 else "#fc8d59" if rain>=1 else "#91bfdb" if rain>0 else "#4575b4"
            folium.Rectangle([[g.min_lat,g.min_lon],[g.max_lat,g.max_lon]],color=color,fill=True,fill_opacity=.5,
                tooltip=f"{g.grid_id} | {rain:.2f} mm | probability {prob:.1%}").add_to(m)
        m.save(p); return p
    except ImportError:
        # No third-party mapping package is required for the basic local grid map.
        lo,hi=grids.longitude_center.min(),grids.longitude_center.max(); la,ha=grids.latitude_center.min(),grids.latitude_center.max()
        sx=max(1.,hi-lo); sy=max(1.,ha-la); shapes=[]
        for _,g in grids.iterrows():
            d=pred.loc[g.grid_id] if pred is not None and g.grid_id in pred.index else None
            rain=float(d.get("predicted_rainfall_mm",0)) if d is not None else 0
            prob=float(d.get("rain_probability",0)) if d is not None else 0
            color="#d73027" if rain>=10 else "#fc8d59" if rain>=1 else "#91bfdb" if rain>0 else "#4575b4"
            x=(g.min_lon-lo)/sx*900+20; y=(hi-g.max_lat)/sy*600+20; w=max(1,(g.max_lon-g.min_lon)/sx*900); h=max(1,(g.max_lat-g.min_lat)/sy*600)
            shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="white"><title>{g.grid_id} | {rain:.2f} mm | probability {prob:.1%}</title></rect>')
        p.write_text('<!doctype html><meta charset="utf-8"><title>Rainfall grid map</title><h2>Rainfall grid map (local schematic)</h2><svg viewBox="0 0 940 640" width="100%" style="background:#f2f5f7">'+''.join(shapes)+'</svg><p>Blue: no rain, light blue: trace, orange: ≥1 mm, red: ≥10 mm. Hover a cell for details.</p>',encoding="utf-8")
        return p
