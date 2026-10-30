"""Authorization regression tests: approved / rejected / review-required are distinct, and nothing is released
unless the decision is an approval. Requests are built the way the app builds them (interpreter + form fields);
no decision or number is hard-coded."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from ehds_hdab.criteria import criteria_to_definitions
from ehds_hdab.decision import decide
from ehds_hdab.pipeline import process_request
from ehds_hdab.providers import interpret

RELEASING = {"APPROVE_STATISTICAL", "APPROVE_SPE"}


def _request(question, purpose, organisation="Test University", requested_as="health data request", record_types=None):
    criteria = dict(interpret(question).criteria)
    if record_types:
        criteria["record_types"] = record_types
    return {
        "id": f"AUTH-{abs(hash((question, purpose, organisation, requested_as))) % 10**6}",
        "applicant": {"name": "Test Applicant", "organisation": organisation, "country": "AT"},
        "requested_as": requested_as,
        "requested_format": "anonymised statistics" if requested_as == "health data request" else "pseudonymised records",
        "purpose": purpose,
        "question": question,
        "definitions": criteria_to_definitions(criteria),
        "criteria": criteria,
    }


def _assert_nothing_released(response):
    assert response["answer"] is None
    assert response["spe"] is None
    assert "withheld" not in response or response["withheld"] is False
    assert not any(key in json.dumps(response.get("answer")) for key in ("patients", "p-"))


def test_insurance_premium_discrimination_is_rejected():
    response = process_request(_request(
        "How many adults have type 2 diabetes, by region?",
        "We are a health insurance company and want to increase insurance premiums for residents of regions with "
        "high diabetes prevalence.", organisation="Example Insurance AG"))
    assert response["decision"] == "REJECT" and response["articles"] == ["54"]
    _assert_nothing_released(response)


def test_targeted_advertising_is_rejected():
    response = process_request(_request(
        "How many adults have obesity, by region?",
        "We plan an advertising campaign for our weight-loss products in regions with the most obesity.",
        organisation="SlimCo GmbH"))
    assert response["decision"] == "REJECT" and response["articles"] == ["54"]
    _assert_nothing_released(response)


def test_specified_scientific_request_is_eligible_for_statistics():
    response = process_request(_request(
        "How many adults have type 2 diabetes, by region?",
        "Scientific research on the regional distribution of type 2 diabetes to plan prevention programmes."))
    assert response["decision"] == "APPROVE_STATISTICAL" and response["articles"] == ["53", "69"]
    assert response["answer"] and response["needs_staff_review"] is False


def test_commercial_organisation_with_specified_research_purpose_is_judged_on_its_merits():
    response = process_request(_request(
        "How many adults have type 2 diabetes, by region?",
        "Scientific research on regional variation in type 2 diabetes care; results will be published in a "
        "peer-reviewed journal.", organisation="GlucoSense GmbH, a private company"))
    assert response["decision"] == "APPROVE_STATISTICAL"
    assert response["answer"]


def test_commercial_organisation_with_unspecified_use_is_held_for_review_and_releases_nothing():
    response = process_request(_request(
        "We are a private company interested in analysing medical records of people with type 2 diabetes for a "
        "commercial project. We have not yet decided how the results will be used.",
        "We want to explore potential commercial applications of diabetes-related health data, but the precise "
        "intended use has not yet been determined.", organisation="Example Private Company"))
    assert response["decision"] == "REVIEW_REQUIRED"
    assert response["articles"] == ["53"] and response["needs_staff_review"] is True
    assert "not matched to a specific Art. 53 category" in response["reason"]
    _assert_nothing_released(response)


@pytest.mark.parametrize("purpose", ["", "   ", "A descriptive analysis for an internal initiative."])
def test_missing_or_unmatched_purpose_is_held_for_review(purpose):
    response = process_request(_request("How many patients have type 2 diabetes?", purpose))
    assert response["decision"] == "REVIEW_REQUIRED"
    _assert_nothing_released(response)


def test_contradictory_purpose_with_prohibited_use_is_rejected_not_approved():
    response = process_request(_request(
        "How many patients have type 2 diabetes?",
        "Scientific research on diabetes, and the insurer will use the results to raise premiums for policyholders."))
    assert response["decision"] == "REJECT"
    _assert_nothing_released(response)


def test_spe_request_with_unresolved_purpose_gets_neither_spe_nor_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr("ehds_hdab.spe.SPE_ROOT", tmp_path)
    response = process_request(_request(
        "Provide pseudonymised conditions and medications for adults with type 2 diabetes.",
        "We want to explore commercial applications of diabetes data; the intended use is not yet determined.",
        organisation="Example Private Company", requested_as="health data access application",
        record_types=["conditions", "medications"]))
    assert response["decision"] == "REVIEW_REQUIRED"
    _assert_nothing_released(response)
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]


def test_spe_request_without_justification_is_downgraded_explicitly_as_the_challenge_requires():
    response = process_request(_request(
        "Provide pseudonymised conditions for adults with type 2 diabetes.",
        "Scientific research on diabetes comorbidity patterns.",
        requested_as="health data access application", record_types=["conditions"]))
    assert response["decision"] == "APPROVE_STATISTICAL"
    assert response["articles"] == ["53", "66", "69"]
    assert "statistics_sufficient" in response["triggered_rules"]
    assert response["spe"] is None


def test_prohibited_use_in_question_with_harmless_form_purpose_is_rejected():
    response = process_request(_request(
        "Which regions have most diabetics so that we can raise insurance premiums there?",
        "Public health research on diabetes."))
    assert response["decision"] == "REJECT" and response["articles"] == ["54"]
    _assert_nothing_released(response)


def test_every_decision_type_respects_the_release_gate():
    cases = [
        ("How many patients have type 2 diabetes?", "Scientific research on diabetes."),
        ("How many patients have type 2 diabetes?", "Commercial project, use not decided."),
        ("How many patients have type 2 diabetes?", "Raise insurance premiums for diabetics."),
    ]
    seen = set()
    for question, purpose in cases:
        response = process_request(_request(question, purpose))
        seen.add(response["decision"])
        if response["decision"] not in RELEASING:
            _assert_nothing_released(response)
        else:
            assert response["answer"] is not None
    assert seen == {"APPROVE_STATISTICAL", "REVIEW_REQUIRED", "REJECT"}


def test_decide_never_returns_an_unknown_decision_type():
    for purpose in ("research", "", "commercial", "advertising our products", "training a prediction model needs "
                    "individual-level data"):
        decision = decide({"purpose": purpose, "question": "How many patients have asthma?",
                           "requested_as": "health data access application"})
        assert decision.decision in RELEASING | {"REJECT", "REVIEW_REQUIRED"}


def test_app_shows_hold_screen_without_numbers_for_unresolved_commercial_purpose():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.text_area(key="research_question").set_value(
        "We are a private company interested in analysing medical records of people with type 2 diabetes for a "
        "commercial project. We have not yet decided how the results will be used.")
    at.button(key="welcome_continue").click()
    at.run()
    while at.session_state["stage"] in ("understand", "clarify"):
        at.button(key=f"{at.session_state['stage']}_continue").click()
        at.run()
    assert at.session_state["stage"] == "review"
    at.text_input(key="review_name").set_value("Test Applicant")
    at.text_input(key="review_org").set_value("Example Private Company")
    at.text_area(key="review_purpose").set_value(
        "We want to explore potential commercial applications of diabetes-related health data, but the precise "
        "intended use has not yet been determined.")
    at.button(key="review_submit").click()
    at.run()
    assert not at.exception
    assert at.session_state["stage"] == "result"
    assert any("REVIEW_REQUIRED" in w.value for w in at.warning)
    assert not at.success and not at.dataframe and not at.metric and not at.download_button
    page = " ".join(m.value for m in at.markdown)
    assert 'class="result-card"' not in page and "1152" not in page and "1,152" not in page
    assert any("No statistics have been computed or released" in m.value for m in at.markdown) or \
        any("No statistics have been computed or released" in str(t.value) for t in at.text)
