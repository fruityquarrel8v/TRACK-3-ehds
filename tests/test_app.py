from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Compare HbA1c levels between male and female patients with type 2 diabetes aged 40-65 during 2025."


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def _advanced_options(at):
    return next(expander for expander in at.expander if expander.label == "Advanced options")


def _submit_example(at, rid="R01"):
    advanced = _advanced_options(at)
    advanced.radio[0].set_value("Example request")
    at.run()
    advanced = _advanced_options(at)
    advanced.selectbox[0].set_value(rid)
    at.run()
    advanced = _advanced_options(at)
    next(button for button in advanced.button if button.label == "Submit this request").click()
    at.run()
    return at


def _fill_review(at):
    at.text_input(key="review_name").set_value("Dr Test")
    at.text_input(key="review_org").set_value("Test University")
    at.text_area(key="review_purpose").set_value("Public health research on diabetes care.")


def test_landing_is_a_single_question_box():
    at = _app()
    assert not at.exception
    assert any("What would you like to research?" in m.value for m in at.markdown)
    assert at.text_area(key="research_question").value == ""
    advanced = _advanced_options(at)
    assert advanced.radio[0].label == "Request source"
    assert "Example request" in advanced.radio[0].options
    assert not any(widget.key == "review_name" for widget in at.text_input)


def test_example_chip_populates_question():
    at = _app()
    next(button for button in at.button if button.label == "Explore diabetes prevalence").click()
    at.run()
    assert "type 2 diabetes" in at.text_area(key="research_question").value


def test_question_flow_interprets_reviews_and_submits():
    at = _app()
    at.text_area(key="research_question").set_value(QUESTION)
    at.button(key="welcome_continue").click()
    at.run()
    assert not at.exception
    assert at.session_state["stage"] == "understand"
    card = next(m.value for m in at.markdown if 'class="card"' in m.value)
    assert "Type 2 diabetes" in card and "ICD-10 E11" in card and "40–65" in card and "Sex" in card
    assert "2025" in card and "HbA1c" in card
    at.button(key="understand_continue").click()
    at.run()
    assert at.session_state["stage"] == "review"
    at.button(key="review_submit").click()
    at.run()
    assert any("Please add" in e.value for e in at.error)
    _fill_review(at)
    at.button(key="review_submit").click()
    at.run()
    assert not at.exception
    assert at.session_state["stage"] == "result"
    assert any("APPROVE_STATISTICAL" in s.value for s in at.success)
    assert any("Your research results" in m.value for m in at.markdown)
    table = at.dataframe[0].value
    assert "7.2" in table.astype(str).to_numpy().ravel().tolist()


def test_unrecognised_concept_cannot_continue():
    at = _app()
    at.text_area(key="research_question").set_value("Number of patients with sepsis by sex")
    at.button(key="welcome_continue").click()
    at.run()
    assert at.session_state["stage"] == "understand"
    assert any("sepsis" in w.value for w in at.warning)
    assert at.error
    assert not any(button.key == "understand_continue" for button in at.button)


def test_medication_clarification_updates_request():
    at = _app()
    at.text_area(key="research_question").set_value(
        "How many adults with atrial fibrillation take an oral anticoagulant, by sex?")
    at.button(key="welcome_continue").click()
    at.run()
    at.button(key="understand_continue").click()
    at.run()
    assert at.session_state["stage"] == "clarify"
    radio = next(r for r in at.radio if r.key.startswith("clarify_"))
    radio.set_value(radio.options[1])
    at.run()
    at.button(key="clarify_continue").click()
    at.run()
    assert at.session_state["stage"] == "review"
    leaves = at.session_state["draft"]["criteria"]["population"]["items"]
    assert any(item.get("type") == "medication" and item.get("status") == "ever" for item in leaves)
    assert not at.session_state["draft"]["assumptions"]


def test_guided_builder_keeps_answers_across_steps():
    at = _app()
    at.button(key="start_guided").click()
    at.run()
    assert at.session_state["stage"] == "guided" and at.session_state["guided_step"] == 1
    at.radio(key="guided_goal").set_value("Compare patient groups")
    at.button(key="guided_next_1").click()
    at.run()
    at.selectbox(key="builder_new_category").set_value("Diagnosis")
    at.button(key="builder_add_criterion").click()
    at.run()
    diagnosis = at.selectbox(key="builder_1_diagnosis")
    t2d = next(option for option in diagnosis.options if option.startswith("Type 2 diabetes"))
    diagnosis.set_value(t2d)
    at.run()
    at.selectbox(key="builder_new_category").set_value("Laboratory measurement")
    at.button(key="builder_add_criterion").click()
    at.run()
    assert at.selectbox(key="builder_1_diagnosis").value == t2d
    assert any("Type 2 diabetes" in i.value for i in at.info)
    at.button(key="builder_2_remove").click()
    at.run()
    assert at.selectbox(key="builder_1_diagnosis").value == t2d
    at.button(key="guided_next_2").click()
    at.run()
    assert at.session_state["guided_step"] == 3
    at.radio(key="guided_analysis").set_value("Compare groups")
    at.run()
    at.multiselect(key="builder_group_by").set_value(["Sex"])
    at.run()
    at.button(key="guided_back_3").click()
    at.run()
    assert at.selectbox(key="builder_1_diagnosis").value == t2d
    at.button(key="guided_back_2").click()
    at.run()
    assert at.radio(key="guided_goal").value == "Compare patient groups"
    at.button(key="guided_next_1").click()
    at.run()
    at.button(key="guided_next_2").click()
    at.run()
    assert at.multiselect(key="builder_group_by").value == ["Sex"]
    at.button(key="guided_next_3").click()
    at.run()
    assert at.session_state["stage"] == "review"
    _fill_review(at)
    at.button(key="review_submit").click()
    at.run()
    assert not at.exception
    assert any("APPROVE_STATISTICAL" in s.value for s in at.success)
    assert at.session_state["last"].startswith("GUIDED-")


def test_edit_details_prefills_guided_builder():
    at = _app()
    at.text_area(key="research_question").set_value(QUESTION)
    at.button(key="welcome_continue").click()
    at.run()
    at.button(key="understand_edit").click()
    at.run()
    assert at.session_state["stage"] == "guided" and at.session_state["guided_step"] == 2
    assert len(at.session_state["builder_criteria"]) >= 2
    assert any(w.key == "builder_1_diagnosis" for w in at.selectbox)


def test_advanced_example_request_and_chart():
    at = _submit_example(_app())
    assert not at.exception
    assert at.session_state["stage"] == "result"
    assert any("APPROVE_STATISTICAL" in s.value for s in at.success)
    chart_input = next(widget for widget in at.text_input if widget.label == "Describe the chart you want")
    chart_input.set_value("bar chart of prevalence by region")
    at.run()
    assert not at.exception
    assert any(button.label == "Download chart (PNG)" for button in at.get("download_button"))
    chart_input = next(widget for widget in at.text_input if widget.label == "Describe the chart you want")
    chart_input.set_value("histogram of HbA1c")
    at.run()
    assert any("secure processing environment" in element.value for element in at.info)


def test_rejected_request_is_explained():
    at = _submit_example(_app(), "R02")
    assert any("Article 54" in m.value or "Art. 54" in m.value for m in at.markdown) or any(
        "Decision on your request" in m.value for m in at.markdown)
    assert not at.dataframe
