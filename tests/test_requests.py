import json
from pathlib import Path
import time

import pandas as pd

from ehds_hdab.data import load_data
from ehds_hdab.pipeline import process_request

ROOT = Path(__file__).resolve().parents[1]


def _expected():
    return json.loads((ROOT / "requests" / "expected.json").read_text(encoding="utf-8"))


def _leaf_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_values(child)
    else:
        yield value


def test_opted_out_patients_are_removed_from_every_loaded_table():
    data = load_data()
    assert not data["patients"]["opted_out"].str.casefold().eq("true").any()
    allowed = set(data["patients"]["patient_id"])
    for table in ("conditions", "medications", "observations", "encounters"):
        assert set(data[table]["patient_id"]).issubset(allowed)
    assert {"region", "data_holder", "sex"} <= set(data["categories"])


def test_decisions_match_all_ten_requests():
    expected = _expected()
    for path in sorted((ROOT / "requests").glob("R*.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        response = process_request(request)
        target = expected[request["id"]]
        assert response["decision"] == target["decision"]
        assert response["articles"] == target["articles"]


def test_expected_values_appear_under_each_group():
    expected = _expected()
    for path in sorted((ROOT / "requests").glob("R*.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        target = expected[request["id"]]
        if target["decision"] == "REJECT":
            continue
        response = process_request(request)
        answer = response["answer"]
        assert isinstance(answer, dict)
        for group, wanted in target["answer"].items():
            assert group in answer, (request["id"], group)
            got_values = list(_leaf_values(answer[group]))
            expected_values = list(_leaf_values(wanted)) if isinstance(wanted, dict) else [wanted]
            for value in expected_values:
                assert any(
                    isinstance(value, (int, float))
                    and isinstance(got, (int, float))
                    and abs(value - got) < 1e-9
                    or value == got
                    for got in got_values
                ), (request["id"], group, value, got_values)


def test_same_request_has_identical_decision_and_answer_three_times():
    request = json.loads((ROOT / "requests" / "R03.json").read_text(encoding="utf-8"))
    outputs = []
    for _ in range(3):
        response = process_request(request)
        outputs.append((response["decision"], response["answer"]))
    assert outputs[0] == outputs[1] == outputs[2]


def test_spe_includes_only_requested_tables_and_hba1c_observations():
    request = json.loads((ROOT / "requests" / "R05.json").read_text(encoding="utf-8"))
    response = process_request(request)
    workspace = Path(response["spe"]["path"])
    assert set(response["spe"]["files"]) == {
        "cohort.csv", "conditions.csv", "medications.csv", "observations.csv"
    }
    observations = pd.read_csv(workspace / "observations.csv", dtype=str)
    assert set(observations["loinc"]) == {"4548-4"}


def test_all_ten_requests_finish_well_under_sixty_seconds():
    started = time.perf_counter()
    for path in sorted((ROOT / "requests").glob("R*.json")):
        process_request(json.loads(path.read_text(encoding="utf-8")))
    assert time.perf_counter() - started < 60
