"""The interpreter must keep requested groupings, measures and access pathway, and must not silently turn
unsupported analyses into plain counts."""
import pytest

from ehds_hdab.decision import decide
from ehds_hdab.nl import interpret_question


def _dims(interp):
    return [g["dimension"] for g in interp.criteria["group_by"]]


@pytest.mark.parametrize("text", [
    "We are a marketing company and want to identify Austrian regions with the highest rates of obesity so that we can "
    "target residents with advertisements for weight-loss products.",
    "Which Austrian regions have the highest rates of obesity?",
    "Compare obesity between regions.",
    "Regional differences in obesity prevalence.",
])
def test_regional_comparison_is_kept(text):
    interp = interpret_question(text)
    assert _dims(interp) == ["region"]
    wants_rate = "rate" in text or "prevalence" in text
    assert any(m["type"] == "percentage" for m in interp.criteria["metrics"]) is wants_rate


def test_region_filter_is_not_mistaken_for_a_regional_grouping():
    interp = interpret_question("How many women with breast cancer are there in Austria, excluding Burgenland?")
    assert _dims(interp) == []


def test_hospital_ranking_is_kept_as_a_hospital_grouping():
    interp = interpret_question("Which hospitals have the most patients with COPD?")
    assert _dims(interp) == ["data_holder"]
    assert interp.unrecognised == []


def test_access_pathway_and_purpose_are_taken_from_the_question():
    interp = interpret_question(
        "We are university researchers studying the relationship between diabetes and cardiovascular disease. We require "
        "access to pseudonymised individual-level records for an approved scientific analysis in a secure processing "
        "environment.")
    assert interp.criteria["access_hint"] == "health data access application"
    assert "university researchers" in interp.criteria["purpose_hint"]
    assert {q["id"] for q in interp.clarifications} == {"diabetes_type", "relationship_conditions"}
    assert any("Cardiovascular disease" == item["label"] for item in interp.criteria["population"]["items"])


def test_plain_clinical_question_has_no_purpose_or_access_hint():
    interp = interpret_question("How many adults have type 2 diabetes, by region?")
    assert "purpose_hint" not in interp.criteria and "access_hint" not in interp.criteria


def test_relationship_between_two_conditions_asks_instead_of_counting_silently():
    interp = interpret_question("Relationship between type 2 diabetes and heart failure, by sex.")
    assert [q["id"] for q in interp.clarifications] == ["relationship_conditions"]
    assert "relationship" not in interp.unrecognised


def test_unsupported_statistical_analysis_is_explained_not_counted():
    interp = interpret_question("Estimate the odds ratio of stroke in patients with atrial fibrillation.")
    assert interp.problems and "odds ratio" in interp.problems[0]
    assert not interp.ok()


def test_hospitalisation_rates_between_regions_keep_grouping_and_state_the_limit():
    interp = interpret_question("Compare hospitalisation rates between regions for heart failure.")
    assert _dims(interp) == ["region"]
    assert any("per 1,000" in a for a in interp.assumptions)
    assert interp.unrecognised == []


def test_marketing_sentence_is_prohibited_from_question_alone_and_with_harmless_form_purpose():
    question = ("We are a marketing company and want to identify Austrian regions with the highest rates of obesity so "
                "that we can target residents with advertisements for weight-loss products.")
    for purpose in ("", "Public health research on obesity."):
        decision = decide({"purpose": purpose, "question": question, "requested_as": "health data request"})
        assert decision.decision == "REJECT" and decision.articles == ["54"]


@pytest.mark.parametrize("text", [
    "Evaluate the impact of alcohol marketing on youth drinking to inform public health policy.",
    "Post-marketing surveillance of SGLT2 inhibitors for public health.",
    "Public health campaign targeting GPs to improve statin prescribing in line with guidelines.",
])
def test_legitimate_marketing_mentions_are_still_not_rejected(text):
    assert decide({"purpose": text, "question": text, "requested_as": "health data request"}).decision != "REJECT"
