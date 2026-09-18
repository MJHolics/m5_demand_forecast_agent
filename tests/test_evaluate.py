import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.evaluate import rmsse, weighted_rmsse


def test_rmsse_perfect_prediction_is_zero():
    train = [10, 12, 14, 16]
    test_actuals = [18, 20]
    test_preds = [18, 20]
    assert rmsse(train, test_actuals, test_preds) == 0.0


def test_rmsse_hand_computed_example():
    # train diffs: 2,2,2 -> denom = mean(4,4,4) = 4
    train = [10, 12, 14, 16]
    test_actuals = [18, 20]
    test_preds = [20, 20]  # errors: 2, 0 -> squared: 4, 0 -> mean/h = 2
    # RMSSE = sqrt(2/4) = sqrt(0.5)
    result = rmsse(train, test_actuals, test_preds)
    assert abs(result - np.sqrt(0.5)) < 1e-9


def test_rmsse_flat_train_series_zero_denominator_with_perfect_pred():
    train = [0, 0, 0, 0]  # naive diffs all 0 -> denom 0
    test_actuals = [0, 0]
    test_preds = [0, 0]
    assert rmsse(train, test_actuals, test_preds) == 0.0


def test_rmsse_flat_train_series_zero_denominator_with_wrong_pred():
    train = [0, 0, 0, 0]
    test_actuals = [0, 0]
    test_preds = [5, 5]
    assert rmsse(train, test_actuals, test_preds) == np.inf


def test_rmsse_trims_leading_zeros_before_launch():
    # 출시 전 0이 섞인 시리즈: [0,0,10,12,14,16] -> 첫 non-zero(10)부터만 diff 계산
    # -> diffs: 2,2,2 -> denom=4, 리딩제로 없는 test_rmsse_hand_computed_example과 동일해야 함
    train_with_leading_zeros = [0, 0, 10, 12, 14, 16]
    train_without = [10, 12, 14, 16]
    test_actuals = [18, 20]
    test_preds = [20, 20]
    assert rmsse(train_with_leading_zeros, test_actuals, test_preds) == \
        rmsse(train_without, test_actuals, test_preds)


def test_rmsse_leading_zeros_inflate_score_when_not_trimmed_would_differ():
    # 리딩제로를 안 자르면 diffs에 (10-0)=10짜리 큰 항이 섞여 denom이 오히려 커진다 —
    # 트리밍 여부에 따라 결과가 실제로 달라진다는 것 자체를 확인(회귀 방지).
    train = [0, 0, 0, 10, 10, 10]  # 트리밍 후: [10,10,10] -> diffs 0,0 -> denom 0
    test_actuals = [10, 10]
    test_preds = [10, 10]
    assert rmsse(train, test_actuals, test_preds) == 0.0  # 완벽 예측 + denom 0 -> 0.0


def test_weighted_rmsse_averages_by_weight_not_evenly():
    rmsse_by_series = {"A": 1.0, "B": 3.0}
    # A가 훨씬 큰 비중을 차지하면 가중평균은 단순평균(2.0)보다 A(1.0)에 가까워야 한다
    weight_by_series = {"A": 0.9, "B": 0.1}
    result = weighted_rmsse(rmsse_by_series, weight_by_series)
    assert result < 2.0
    assert abs(result - (1.0 * 0.9 + 3.0 * 0.1)) < 1e-9
