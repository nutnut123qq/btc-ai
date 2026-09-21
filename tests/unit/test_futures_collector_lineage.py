from datetime import datetime, timezone

from futures_collector import upsert_rows


class RecordingCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))


def test_poll_window_marks_old_rows_reconstructed_but_keeps_recent_row_live():
    cursor = RecordingCursor()
    received = datetime.fromtimestamp(1_000, timezone.utc)

    upsert_rows(
        cursor,
        "BTCUSDT",
        [(990_000, 1.0), (100_000, 2.0)],
        {"a": "GlobalLsRatio"},
        source="fixture",
        reconstructed=False,
        received_at_utc=received,
        live_max_lag_ms=15_000,
    )

    assert cursor.calls[0][1][-1] is False
    assert cursor.calls[1][1][-1] is True


def test_repoll_sql_preserves_first_availability_when_no_new_field_arrives():
    cursor = RecordingCursor()
    upsert_rows(
        cursor,
        "BTCUSDT",
        [(990_000, 1.0)],
        {"a": "GlobalLsRatio"},
        source="fixture",
        reconstructed=False,
        received_at_utc=datetime.fromtimestamp(1_000, timezone.utc),
    )

    sql = cursor.calls[0][0]
    assert '"GlobalLsRatio" IS NULL AND EXCLUDED."GlobalLsRatio" IS NOT NULL' in sql
    assert 'ELSE "FuturesMetrics"."AvailableTimeMs" END' in sql
    assert 'ELSE "FuturesMetrics"."IsReconstructed" END' in sql
