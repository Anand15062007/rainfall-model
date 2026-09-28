"""Append reproducible experiment records as JSONL and tabular CSV."""
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def log_experiment(record, directory="experiments"):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    record={"experiment_id":record.get("experiment_id",str(uuid.uuid4())),"created_at":datetime.now(timezone.utc).isoformat(),**record}
    with (root/"runs.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(record,default=str,allow_nan=True)+"\n")
    rows=[]; p=root/"runs.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines(): rows.append(json.loads(line))
    pd.json_normalize(rows).to_csv(root/"runs.csv",index=False)
    return record
