from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _app():
    return AppTest.from_file(str(ROOT / "app.py")).run()


def _advanced_options(at):
    return next(expander for expander in at.expander if expander.label == "Advanced options")


def test_applicant_page_runs_without_exception():
    at = _app()

    assert not at.exception


def test_advanced_options_contains_request_source_radio():
    at = _app()
    advanced = _advanced_options(at)

    assert len(advanced.radio) == 1
    assert advanced.radio[0].label == "Request source"
    assert advanced.radio[0].options == ["Example request", "Upload JSON", "Paste JSON"]


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
    next(widget for widget in at.text_area if widget.label == "Purpose").set_value(
        "Public health research on type 2 diabetes prevalence."
    )
    next(widget for widget in at.text_area if widget.label == "Question").set_value(
        "How many adults have type 2 diabetes, by sex?"
    )
    next(widget for widget in at.text_input if widget.label == "Applicant name").set_value(
        "Test Applicant"
    )
    next(widget for widget in at.text_input if widget.label == "Organisation").set_value(
        "Test University"
    )
    next(button for button in at.button if button.label == "Submit request").click()
    at.run()

    assert not at.exception
    assert any("APPROVE_STATISTICAL" in element.value for element in at.success)
