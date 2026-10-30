from decimal import Decimal

from ehds_hdab.privacy import apply_small_cell, check_statistical_output, round_half_up


def _raw(value, kind, **metadata):
    metric = {"id": "m", "label": "measure", "type": kind}
    result = {"type": kind, "value": value, **metadata}
    return {"metrics": [metric], "groups": [{"path": ["Group"], "values": {"m": result}}]}


def test_counts_suppress_below_ten_including_zero():
    assert apply_small_cell(_raw(9, "count_patients"))["Group"]["measure"] == "<10"
    assert apply_small_cell(_raw(0, "count_encounters"))["Group"]["measure"] == "<10"
    assert apply_small_cell(_raw(10, "count_patients"))["Group"]["measure"] == 10


def test_percentage_and_mean_small_cells_are_suppressed():
    assert apply_small_cell(
        _raw(Decimal("45"), "percentage", numerator=9, denominator=20)
    )["Group"]["measure"] == "suppressed"
    assert apply_small_cell(
        _raw(Decimal("12.5"), "mean", n=9)
    )["Group"]["measure"] == "suppressed"


def test_round_half_up_and_trailing_decimal():
    assert round_half_up(Decimal("0.25")) == Decimal("0.3")
    assert apply_small_cell(
        _raw(Decimal("37"), "percentage", numerator=37, denominator=100)
    )["Group"]["measure"] == 37.0


def test_statistical_output_check_flags_ids_small_counts_and_records():
    assert check_statistical_output({"x": 9, "y": "p-abcdef12", "z": [{"value": 11}]})
