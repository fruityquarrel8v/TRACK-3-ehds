from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _app():
    return AppTest.from_file(str(ROOT / "app.py")).run()


def _advanced_options(at):
    return next(expander for expander in at.expander if expander.label == "Advanced options")


def _describe_tab(at):
    return next(tab for tab in at.tabs if tab.label == "Describe your research")


def _builder_tab(at):
    return next(tab for tab in at.tabs if tab.label == "Build your research cohort")


def test_applicant_page_runs_without_exception():
    at = _app()

    assert not at.exception


def test_advanced_options_contains_request_source_radio():
    at = _app()
    advanced = _advanced_options(at)

    assert len(advanced.radio) == 1
    assert advanced.radio[0].label == "Request source"
    assert advanced.radio[0].options == [
        "Example request",
        "Upload JSON",
        "Paste JSON",
        "Fill in a form (challenge format)",
    ]


def test_advanced_example_request_submission_shows_statistical_decision():
    at = _app()
    advanced = _advanced_options(at)
    advanced.radio[0].set_value("Example request")
    at.run()

    advanced = _advanced_options(at)
    advanced.selectbox[0].set_value("R01")
    at.run()

    advanced = _advanced_options(at)
    next(button for button in advanced.button if button.label == "Submit this request").click()
    at.run()

    assert not at.exception
    assert any("APPROVE_STATISTICAL" in element.value for element in at.success)


def test_form_submission_shows_statistical_decision():
    at = _app()
    describe = _describe_tab(at)
    next(widget for widget in describe.text_area if widget.label == "Purpose *").set_value(
        "Public health research on type 2 diabetes prevalence."
    )
    next(widget for widget in describe.text_area if widget.label == "Research question").set_value(
        "How many adults have type 2 diabetes, by sex?"
    )
    next(widget for widget in describe.text_input if widget.label == "Applicant name").set_value(
        "Test Applicant"
    )
    next(widget for widget in describe.text_input if widget.label == "Organisation").set_value(
        "Test University"
    )
    next(button for button in describe.button if button.label == "Interpret my question").click()
    at.run()
    describe = _describe_tab(at)
    next(button for button in describe.button if button.label == "Confirm and submit").click()
    at.run()

    assert not at.exception
    assert any("APPROVE_STATISTICAL" in element.value for element in at.success)


def test_research_cohort_builder_adds_diagnosis_and_submits():
    at = _app()
    builder = _builder_tab(at)
    next(widget for widget in builder.text_area if widget.label == "Purpose *").set_value(
        "Public health research on type 2 diabetes prevalence."
    )
    next(widget for widget in builder.selectbox if widget.label == "Criterion type").set_value(
        "Diagnosis"
    )
    next(button for button in builder.button if button.label == "Add criterion").click()
    at.run()

    builder = _builder_tab(at)
    next(widget for widget in builder.selectbox if widget.label == "Diagnosis").set_value(
        "Type 2 diabetes (ICD-10 E11)"
    )
    next(button for button in builder.button if button.label == "Submit request").click()
    at.run()

    assert not at.exception
    assert any("APPROVE_STATISTICAL" in element.value for element in at.success)


def test_builder_lab_threshold_default_tracks_selected_loinc():
    at = _app()
    builder = _builder_tab(at)
    from_year = next(widget for widget in builder.selectbox if widget.label == "From year")
    assert "1995" in from_year.options
    next(widget for widget in builder.selectbox if widget.label == "Criterion type").set_value(
        "Laboratory measurement"
    )
    next(button for button in builder.button if button.label == "Add criterion").click()
    at.run()

    builder = _builder_tab(at)
    lab = next(widget for widget in builder.selectbox if widget.label == "Laboratory measurement")
    hba1c = next(option for option in lab.options if "4548-4" in option)
    lab.set_value(hba1c)
    at.run()
    builder = _builder_tab(at)
    assert next(widget for widget in builder.number_input if widget.label == "Threshold (%)").value == 8.0

    lab = next(widget for widget in builder.selectbox if widget.label == "Laboratory measurement")
    bmi = next(option for option in lab.options if "39156-5" in option)
    lab.set_value(bmi)
    at.run()
    builder = _builder_tab(at)

    assert next(
        widget for widget in builder.number_input if widget.label == "Threshold (kg/m2)"
    ).value == 30.0


def _nl_tab(at):
    return next(tab for tab in at.tabs if tab.label == "Describe your research")


def _interpret_nl_example(at):
    describe = _nl_tab(at)
    next(widget for widget in describe.text_area if widget.label == "Purpose *").set_value(
        "Public health research on diabetes outcomes."
    )
    next(widget for widget in describe.text_area if widget.label == "Research question").set_value(
        "Compare HbA1c levels between male and female patients with type 2 diabetes "
        "aged 40-65 during 2025."
    )
    next(button for button in describe.button if button.label == "Interpret my question").click()
    return at.run()


def test_natural_language_interpretation_and_submission():
    at = _interpret_nl_example(_app())

    assert not at.exception
    assert any(
        "hba1c" in element.value.casefold()
        and "type 2 diabetes" in element.value.casefold()
        for element in at.info
    )
    next(button for button in at.button if button.label == "Confirm and submit").click()
    at.run()

    assert not at.exception
    assert any("APPROVE_STATISTICAL" in element.value for element in at.success)


def test_natural_language_correction_prefills_builder_rows():
    at = _interpret_nl_example(_app())
    next(button for button in at.button if button.label == "Correct in cohort builder").click()
    at.run()

    assert not at.exception
    assert len(at.session_state["builder_criteria"]) >= 2


def test_since_condition_timing_survives_cohort_builder_correction():
    at = _app()
    describe = _nl_tab(at)
    next(widget for widget in describe.text_area if widget.label == "Purpose *").set_value(
        "Public health research on asthma."
    )
    next(widget for widget in describe.text_area if widget.label == "Research question").set_value(
        "Patients diagnosed with asthma since 2022"
    )
    next(button for button in describe.button if button.label == "Interpret my question").click()
    at.run()
    next(button for button in at.button if button.label == "Correct in cohort builder").click()
    at.run()

    builder = next(tab for tab in at.tabs if tab.label == "Build your research cohort")
    diagnosis_timing = next(
        radio for radio in builder.radio if radio.label == "Diagnosis timing"
    )
    assert diagnosis_timing.value == "Diagnosed since the start of the time period"
    assert next(
        selectbox for selectbox in builder.selectbox if selectbox.label == "From year"
    ).value == 2022
    assert not at.exception


def test_unrecognised_question_disables_natural_language_submission():
    at = _app()
    describe = _nl_tab(at)
    next(widget for widget in describe.text_area if widget.label == "Purpose *").set_value(
        "Public health research on infections."
    )
    next(widget for widget in describe.text_area if widget.label == "Research question").set_value(
        "Number of patients with sepsis by sex"
    )
    next(button for button in describe.button if button.label == "Interpret my question").click()
    at.run()

    assert not at.exception
    assert any(
        "Nothing clinical or demographic was recognised" in element.value
        for element in at.error
    )
    submit = next(button for button in at.button if button.label == "Confirm and submit")
    assert submit.disabled
