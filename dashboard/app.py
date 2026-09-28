"""Streamlit local dashboard for generated artifacts and offline observations."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="City Rainfall",layout="wide")
st.title("City-Level Rainfall Prediction")
st.caption("Local research dashboard. Synthetic data and model scores are demonstrations, not operational forecasts.")
weather=Path("data/synthetic/weather.csv"); grids=Path("data/processed/grids.csv"); metrics=Path("experiments/latest_metrics.json")
if not weather.exists():
    st.info("Generate data and run the pipeline first: `python -m app.pipeline`")
    st.stop()
df=pd.read_csv(weather,parse_dates=["timestamp"]); g=pd.read_csv(grids) if grids.exists() else pd.DataFrame()
tab1,tab2,tab3,tab4=st.tabs(["Overview","Grid map","Model comparison","Historical rainfall"])
with tab1:
    a,b,c=st.columns(3); a.metric("City",g.city.iloc[0] if len(g) else "—"); b.metric("Grids",len(g)); c.metric("Observations",len(df))
    st.dataframe(df.sort_values("timestamp").tail(20),use_container_width=True)
with tab2:
    path=Path("data/processed/grid_map.html")
    if path.exists(): st.components.v1.html(path.read_text(encoding="utf-8"),height=650,scrolling=True)
    else: st.info("Run the pipeline to create grid rainfall map.")
with tab3:
    if metrics.exists():
        scores=json.loads(metrics.read_text()); st.dataframe(pd.DataFrame([{"model":k,**v} for k,v in scores.items()]).set_index("model"),use_container_width=True)
    else: st.info("No evaluations found.")
with tab4:
    col=st.selectbox("Variable",["rainfall_1h","temperature","relative_humidity","cloud_cover"])
    df.groupby("timestamp")[col].mean().plot(title=f"Mean {col}")
    st.pyplot(__import__("matplotlib.pyplot",fromlist=["pyplot"]).gcf()); __import__("matplotlib.pyplot",fromlist=["pyplot"]).close()
