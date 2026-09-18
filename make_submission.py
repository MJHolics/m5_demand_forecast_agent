"""공식 Kaggle 제출 파일 생성. 이번 세션에서 seasonal-naive가 LightGBM 5종을 전부 이겨서,
현재 최선 모델(naive)로 제출한다 — 외부(Private LB)에서도 같은 결론이 나오는지 검증하는 용도.
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
from m5fc.data_loader import build_long_table, load_calendar
from m5fc.models import seasonal_naive_forecast

long_df = build_long_table("data")
cal = load_calendar("data")

train_end_val = cal.loc[cal["d"] == "d_1913", "date"].iloc[0]
train_end_eval = cal.loc[cal["d"] == "d_1941", "date"].iloc[0]

val_pred = seasonal_naive_forecast(long_df, train_end_val, horizon=28)
eval_pred = seasonal_naive_forecast(long_df, train_end_eval, horizon=28)

def to_submission_rows(pred_wide: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = pred_wide.copy()
    out.columns = [f"F{i}" for i in range(1, len(out.columns) + 1)]
    out.index = [i.replace("_evaluation", f"_{suffix}") for i in out.index]
    out.index.name = "id"
    return out.reset_index()

val_rows = to_submission_rows(val_pred, "validation")
eval_rows = to_submission_rows(eval_pred, "evaluation")

submission = pd.concat([val_rows, eval_rows], ignore_index=True)

sample = pd.read_csv("data/sample_submission.csv")
submission = sample[["id"]].merge(submission, on="id", how="left")
assert submission.isna().sum().sum() == 0, "매칭 안 된 id가 있다 — id 접미사 규칙을 다시 확인할 것"
assert submission.shape == sample.shape, f"shape mismatch: {submission.shape} vs {sample.shape}"

submission.to_csv("submission.csv", index=False)
print("saved submission.csv", submission.shape)
