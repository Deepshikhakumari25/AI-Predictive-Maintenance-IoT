import json

from predictive_maintenance.utils import append_jsonl, load_recent_events


def test_event_history_encodes_non_finite_sensor_values_as_json_null(tmp_path):
    path = tmp_path / "history.jsonl"
    append_jsonl(path, {"reading": {"temperature": float("nan")}})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["reading"]["temperature"] is None


def test_recent_event_history_loads_only_requested_tail(tmp_path):
    path = tmp_path / "history.jsonl"
    for number in range(5):
        append_jsonl(path, {"number": number})
    assert load_recent_events(path, limit=2) == [{"number": 3}, {"number": 4}]
