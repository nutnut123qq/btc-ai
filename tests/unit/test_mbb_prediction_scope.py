"""MBB-04D: prediction stays fail-closed for MBB — no BTC artifact can serve it.

The model registry keys artifacts as ``{symbol}_{timeframe}_ws{w}_h{h}`` and
the manifest identity check requires ``manifest["symbol"] == requested``;
``require_active_symbol`` additionally keeps the known-but-disabled MBB out of
production inference until a separately trained, gated artifact exists.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import prediction_service
from prediction_service import ModelArtifactIncompatibleError


def test_mbb_prediction_rejected_at_symbol_gate():
    with pytest.raises(ValueError, match="Unsupported symbol"):
        prediction_service._resolve_active_artifact("MBB", "1d", 15, "1d")


def test_btc_registry_keys_cannot_answer_mbb():
    # No MBB_* key exists in the registry, and a BTCUSDT manifest could never
    # satisfy the identity check (manifest["symbol"] == requested symbol) even
    # if a rogue MBB_* entry were registered.
    keys = [k for k in prediction_service._registry_models() if k.startswith("MBB_")]
    assert keys == []
    entry = prediction_service._registry_models().get("MBB_1d_ws15_h1d")
    assert not isinstance(entry, dict) or entry.get("status") != "active"


def test_btcusdt_path_still_resolves_or_fails_on_artifact_not_scope():
    # BTC requests must not be rejected by the scope layer; they may still fail
    # on a missing artifact when no model is registered locally.
    try:
        prediction_service._resolve_active_artifact("BTCUSDT", "4h", 15, "1d")
    except ModelArtifactIncompatibleError as exc:
        assert "BTCUSDT" in str(exc)
    except ValueError as exc:  # scope must never be the failure reason
        pytest.fail(f"BTC scope rejected unexpectedly: {exc}")
