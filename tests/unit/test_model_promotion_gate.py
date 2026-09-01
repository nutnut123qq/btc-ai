import copy

from rolling_retrainer import assess_promotion_gate


def _metrics():
    return {
        "samples": 180,
        "class_counts": [30, 120, 30],
        "f1_per_class": {"down": 0.45, "sideways": 0.80, "up": 0.44},
        "f1_macro": 0.56,
        "balanced_accuracy": 0.58,
        "mcc": 0.35,
        "ece": 0.08,
        "brier_score": 0.40,
        "log_loss": 0.72,
    }


def _baseline():
    return {"brier_score": 0.50, "log_loss": 0.90}


def test_gate_requires_both_independent_windows_to_pass():
    assert assess_promotion_gate(_metrics(), _metrics(), _baseline(), _baseline())["passed"]

    collapsed = copy.deepcopy(_metrics())
    collapsed.update({"f1_macro": 0.29, "balanced_accuracy": 0.33, "mcc": 0.0})
    collapsed["f1_per_class"] = {"down": 0.0, "sideways": 0.87, "up": 0.0}
    result = assess_promotion_gate(_metrics(), collapsed, _baseline(), _baseline())
    assert not result["passed"]
    assert any(failure.startswith("oos:") for failure in result["failures"])


def test_gate_fails_closed_when_metrics_or_class_support_are_missing():
    assert not assess_promotion_gate({}, _metrics(), _baseline(), _baseline())["passed"]

    sparse = copy.deepcopy(_metrics())
    sparse["class_counts"] = [4, 170, 6]
    result = assess_promotion_gate(sparse, _metrics(), _baseline(), _baseline())
    assert not result["passed"]
    assert "validation: class support" in result["failures"]
