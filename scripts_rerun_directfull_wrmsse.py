"""2026-09-17: leading-zero 트리밍 수정 후 LightGBM(직접·전체이력 5.2년) WRMSSE 재계산.
기존 최고 기록(버그 있던 공식) 0.9386과 비교. 예측을 parquet로 저장해 재평가 재사용 가능하게 한다."""
import sys
import time

sys.path.insert(0, ".")
from m5fc.data_loader import build_long_table, load_calendar
from m5fc.models_gbm import train_direct_multihorizon, direct_multihorizon_forecast
from m5fc.evaluate import full_wrmsse

t0 = time.time()
long_df = build_long_table("data")
cal = load_calendar("data")
train_end_val = cal.loc[cal["d"] == "d_1913", "date"].iloc[0]

print(f"train_end={train_end_val}, min_train_date=None(전체이력), load elapsed={time.time()-t0:.1f}s", flush=True)

t1 = time.time()
models = train_direct_multihorizon(long_df, train_end_val, horizon=28, min_train_date=None)
print(f"train elapsed={time.time()-t1:.1f}s", flush=True)

t2 = time.time()
preds = direct_multihorizon_forecast(models, long_df, train_end_val, horizon=28)
print(f"predict elapsed={time.time()-t2:.1f}s", flush=True)
preds.to_parquet("artifacts/pred_direct_fullhistory.parquet")
print("saved artifacts/pred_direct_fullhistory.parquet", flush=True)

t3 = time.time()
result = full_wrmsse(long_df, preds, train_end_val, horizon=28)
print(f"eval elapsed={time.time()-t3:.1f}s", flush=True)

print("OVERALL:", result["overall"])
for k, v in result["by_level"].items():
    print(f"  {k}: {v:.4f}")
print(f"TOTAL elapsed {time.time()-t0:.1f}s", flush=True)
print("DONE_MARKER", flush=True)
