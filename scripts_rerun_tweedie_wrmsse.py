"""2026-09-17: leading-zero 트리밍 수정 후 LightGBM(직접·180일·Tweedie objective) WRMSSE 재계산.
기존 최고 기록(버그 있던 공식) 0.9374와 비교. tweedie_variance_power=1.1, n_estimators=150(기존 설정 그대로)."""
import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from m5fc.data_loader import build_long_table, load_calendar
from m5fc.models_gbm import train_direct_multihorizon, direct_multihorizon_forecast
from m5fc.evaluate import full_wrmsse

t0 = time.time()
long_df = build_long_table("data")
cal = load_calendar("data")
train_end_val = cal.loc[cal["d"] == "d_1913", "date"].iloc[0]

min_train_date = train_end_val - pd.Timedelta(days=179)
print(f"train_end={train_end_val}, min_train_date={min_train_date}, load elapsed={time.time()-t0:.1f}s", flush=True)

t1 = time.time()
models = train_direct_multihorizon(
    long_df, train_end_val, horizon=28, min_train_date=min_train_date,
    objective="tweedie", tweedie_variance_power=1.1, n_estimators=150,
)
print(f"train elapsed={time.time()-t1:.1f}s", flush=True)

t2 = time.time()
preds = direct_multihorizon_forecast(models, long_df, train_end_val, horizon=28)
print(f"predict elapsed={time.time()-t2:.1f}s", flush=True)
preds.to_parquet("artifacts/pred_tweedie.parquet")
print("saved artifacts/pred_tweedie.parquet", flush=True)

t3 = time.time()
result = full_wrmsse(long_df, preds, train_end_val, horizon=28)
print(f"eval elapsed={time.time()-t3:.1f}s", flush=True)

print("OVERALL:", result["overall"])
for k, v in result["by_level"].items():
    print(f"  {k}: {v:.4f}")
print(f"TOTAL elapsed {time.time()-t0:.1f}s", flush=True)
print("DONE_MARKER", flush=True)
