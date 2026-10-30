"""Successive requests in one browser session must not share purpose, access type, clarifications or decisions."""
from streamlit.testing.v1 import AppTest

COMMERCIAL_Q = ("We are a private company interested in analysing medical records of people with type 2 diabetes for a "
                "commercial project. We have not yet decided how the results will be used.")
COMMERCIAL_PURPOSE = ("We want to explore potential commercial applications of diabetes-related health data, but the "
                      "precise intended use has not yet been determined.")
ADVERT_Q = ("We are a marketing company and want to identify Austrian regions with the highest rates of obesity so that "
            "we can target residents with advertisements for weight-loss products.")
RESEARCH_Q = "How many adults have type 2 diabetes, by region?"
SPE_Q = ("We are university researchers studying the relationship between diabetes and cardiovascular disease. We require "
         "access to pseudonymised individual-level records for an approved scientific analysis in a secure processing "
         "environment.")
SPE_OPTION = ("Access to pseudonymised records in a secure processing environment (health data access application)")


def _app():
    return AppTest.from_file("../app.py", default_timeout=120).run()


def _to_review(at, question):
    at.text_area(key="research_question").set_value(question)
    at.button(key="welcome_continue").click()
    at.run()
    assert not at.exception
    for _ in range(3):
        stage = at.session_state["stage"]
        if stage == "review":
            break
        assert stage in ("understand", "clarify"), stage
        at.button(key=f"{stage}_continue").click()
        at.run()
        assert not at.exception
    assert at.session_state["stage"] == "review"


def _summary_card(at):
    return next(m.value for m in at.markdown if "What will be assessed" in m.value)


def _submit(at):
    at.button(key="review_submit").click()
    at.run()
    assert not at.exception
    assert at.session_state["stage"] == "result"
    return at.session_state["decisions"][at.session_state["last"]], at.session_state["requests"][at.session_state["last"]]


def test_purpose_access_type_and_decision_do_not_carry_over_between_requests():
    at = _app()
    _to_review(at, COMMERCIAL_Q)
    at.text_input(key="review_name").set_value("Test Applicant")
    at.text_input(key="review_org").set_value("Example Private Company")
    at.text_area(key="review_purpose").set_value(COMMERCIAL_PURPOSE)
    at.selectbox(key="review_request_type").set_value(SPE_OPTION)
    at.run()
    assert COMMERCIAL_PURPOSE[:40] in _summary_card(at)
    first, first_request = _submit(at)
    assert first["decision"] == "REVIEW_REQUIRED" and first["answer"] is None
    assert first_request["purpose"] == COMMERCIAL_PURPOSE
    assert first_request["requested_as"] == "health data access application"

    at.button(key="result_new").click()
    at.run()
    assert "last" not in at.session_state and at.session_state["draft"] == {}
    _to_review(at, ADVERT_Q)
    purpose_box = at.text_area(key="review_purpose").value or ""
    assert "commercial" not in purpose_box.casefold()
    assert "marketing company" in purpose_box.casefold()          # pre-filled from the new question, visibly
    assert at.selectbox(key="review_request_type").value != SPE_OPTION
    assert at.text_input(key="review_name").value == "Test Applicant"  # identity is kept
    card = _summary_card(at)
    assert "marketing company" in card.casefold() and "commercial applications" not in card.casefold()
    second, second_request = _submit(at)
    assert second_request["purpose"] == purpose_box and second_request["requested_as"] == "health data request"
    assert second["decision"] == "REJECT" and second["articles"] == ["54"] and second["answer"] is None
    assert not at.dataframe and not at.download_button

    at.button(key="result_new").click()
    at.run()
    _to_review(at, RESEARCH_Q)
    assert (at.text_area(key="review_purpose").value or "") == ""   # no purpose cue in a plain clinical question
    at.text_area(key="review_purpose").set_value("Scientific research on the regional distribution of type 2 diabetes.")
    third, third_request = _submit(at)
    assert third["decision"] == "APPROVE_STATISTICAL" and third["answer"]
    assert third_request["purpose"].startswith("Scientific research")
    assert third_request["question"] == RESEARCH_Q
    assert len({first["request_id"], second["request_id"], third["request_id"]}) == 3


def test_spe_request_keeps_access_pathway_and_relationship_from_the_question():
    at = _app()
    _to_review(at, SPE_Q)
    assert at.selectbox(key="review_request_type").value == SPE_OPTION
    assert "university researchers" in (at.text_area(key="review_purpose").value or "").casefold()
    at.text_input(key="review_name").set_value("Dr Test")
    at.text_input(key="review_org").set_value("Medical University of Vienna")
    at.multiselect(key="review_record_types").set_value(["conditions", "medications"])
    at.run()
    card = _summary_card(at)
    assert "secure processing environment" in card.casefold() and "conditions, medications" in card
    assert "Type 2 diabetes" in card and "Cardiovascular disease" in card
    resp, request = _submit(at)
    assert request["requested_as"] == "health data access application"
    assert resp["decision"] == "APPROVE_SPE" and resp["articles"] == ["53", "66", "68", "73"]
    assert resp["spe"]["cohort_size"] >= 10 and resp["answer"] == {"cohort_size": resp["spe"]["cohort_size"]}
