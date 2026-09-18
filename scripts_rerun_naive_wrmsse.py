"""2026-09-17: leading-zero 트리밍 수정 후 naive 베이스라인 WRMSSE 재계산 (0.9128과 비교용)."""
import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from m5fc.data_loader import build_long_table, load_calendar
from m5fc.models import seasonal_naive_forecast
from m5fc.evaluate import full_wrmsse

t0 = time.time()
long_df = build_long_table("data")
cal = load_calendar("data")
train_end_val = cal.loc[cal["d"] == "d_1913", "date"].iloc[0]

pred = seasonal_naive_forecast(long_df, train_end_val, horizon=28)
result = full_wrmsse(long_df, pred, train_end_val, horizon=28)

print("OVERALL:", result["overall"])
for k, v in result["by_level"].items():
    print(f"  {k}: {v:.4f}")
print(f"elapsed {time.time()-t0:.1f}s")
