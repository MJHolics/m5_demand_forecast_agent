import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.forecast_store import save_artifact, load_artifact


def test_save_and_load_artifact_roundtrip(tmp_path):
    artifact = {
        "model": "seasonal_naive_7day",
        "train_end": "2016-04-24",
        "forecast_dates": ["2016-04-25", "2016-04-26"],
        "events_in_forecast_window": [],
        "snap_days_by_state": {"CA": [], "TX": [], "WI": []},
        "series": {
            "FOODS_1_001_CA_1_evaluation": {
                "item_id": "FOODS_1_001", "store_id": "CA_1", "cat_id": "FOODS",
                "dept_id": "FOODS_1", "state_id": "CA",
                "forecast": {"2016-04-25": 3.0, "2016-04-26": 4.0},
                "recent_actual": {"2016-04-24": 2.0},
                "current_price": 2.5,
            }
        },
    }
    path = tmp_path / "artifact.json"
    save_artifact(artifact, str(path))
    loaded = load_artifact(str(path))
    assert loaded == artifact
    assert loaded["series"]["FOODS_1_001_CA_1_evaluation"]["forecast"]["2016-04-26"] == 4.0
