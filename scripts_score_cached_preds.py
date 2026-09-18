"""2026-09-18: 어젯밤 fork가 재학습까지는 끝냈지만(예측 parquet 저장) 채점·README 반영 전에
중단됨. 재학습 없이 캐시된 예측만 불러와 수정된 rmsse()로 채점 마무리."""
import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from m5fc.data_loader import build_long_table, load_calendar
from m5fc.evaluate import full_wrmsse

t0 = time.time()
long_df = build_long_table("data")
cal = load_calendar("data")
train_end_val = cal.loc[cal["d"] == "d_1913", "date"].iloc[0]
print(f"load elapsed={time.time()-t0:.1f}s", flush=True)

for name, path in [
    ("direct_fullhistory", "artifacts/pred_direct_fullhistory.parquet"),
    ("tweedie", "artifacts/pred_tweedie.parquet"),
]:
    preds = pd.read_parquet(path)
    t1 = time.time()
    result = full_wrmsse(long_df, preds, train_end_val, horizon=28)
    print(f"=== {name} (eval elapsed={time.time()-t1:.1f}s) ===")
    print("OVERALL:", result["overall"])
    for k, v in result["by_level"].items():
        print(f"  {k}: {v:.4f}")

print(f"TOTAL elapsed {time.time()-t0:.1f}s", flush=True)
print("DONE_MARKER", flush=True)
