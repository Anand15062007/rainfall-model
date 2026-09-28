"""Small SQLite persistence layer with a clean SQLAlchemy optional path."""
import sqlite3
from pathlib import Path
import pandas as pd

class Store:
    TABLES=("grids","weather_observations","predictions","experiments","model_runs","api_logs")
    def __init__(self,path="data/rainfall.sqlite"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS grids (grid_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS weather_observations (timestamp TEXT, grid_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(timestamp,grid_id));
            CREATE TABLE IF NOT EXISTS predictions (timestamp TEXT, grid_id TEXT, model TEXT, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_runs (model_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS api_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS ix_weather_time ON weather_observations(timestamp);
            """)
    def put_frame(self,table,frame):
        if table not in {"grids","weather_observations","predictions"}: raise ValueError("unsupported frame table")
        df=frame.copy();
        with sqlite3.connect(self.path) as c:
            for _,r in df.iterrows():
                payload=r.to_json(date_format="iso")
                if table=="grids": c.execute("INSERT OR REPLACE INTO grids VALUES (?,?)",(str(r.grid_id),payload))
                elif table=="weather_observations": c.execute("INSERT OR REPLACE INTO weather_observations VALUES (?,?,?)",(str(r.timestamp),str(r.grid_id),payload))
                else: c.execute("INSERT INTO predictions(timestamp,grid_id,model,payload) VALUES (?,?,?,?)",(str(r.timestamp),str(r.grid_id),str(r.get('model','')),payload))
