"""Cloudflare Pages 배포 준비: forecast_store.json(39MB, 단일파일)을 주(state) 3개로 쪼갠다.
Cloudflare Pages 정적 자산 파일당 상한은 25 MiB(공식 문서 확인, 2026-09-17) -- 단일 파일로는
못 올린다. 메타데이터(model/train_end/forecast_dates/events/snap)는 각 샤드에 중복 포함해
클라이언트가 샤드 하나만 받아도 완결된 응답을 만들 수 있게 한다.
"""
import json
from pathlib import Path

SRC = Path("artifacts/forecast_store.json")
OUT_DIR = Path("artifacts/cf_shards")
OUT_DIR.mkdir(exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    store = json.load(f)

meta = {k: v for k, v in store.items() if k != "series"}
series = store["series"]

by_state: dict[str, dict] = {"CA": {}, "TX": {}, "WI": {}}
for series_id, payload in series.items():
    # id 형식: <item_id>_<store_id>_evaluation, store_id는 CA_1/TX_2/WI_3 형태
    state = series_id.split("_")[-3] if series_id.endswith("_evaluation") else None
    if state not in by_state:
        raise ValueError(f"알 수 없는 state 접두: {series_id}")
    by_state[state][series_id] = payload

MB = 1024 * 1024
LIMIT = 25 * MB
for state, subset in by_state.items():
    out = dict(meta)
    out["series"] = subset
    out_path = OUT_DIR / f"forecast_{state}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size = out_path.stat().st_size
    status = "OK" if size < LIMIT else "초과!!"
    print(f"{state}: {len(subset)}개 시리즈, {size/MB:.1f}MB ({status}, 상한 25MB)")
