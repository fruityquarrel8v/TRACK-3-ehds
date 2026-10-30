"""Streamlit web interface for the simulated EHDS Health Data Access Body.

Run: streamlit run app.py
Patient-level data never leaves this machine; no external services are called.
"""
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from ehds_hdab import config
from ehds_hdab.criteria import criteria_to_definitions, describe_criteria
from ehds_hdab.data import load_data
from ehds_hdab.pipeline import process_request
from ehds_hdab.spe import DownloadBlockedError, SecureProcessingEnvironment
from ehds_hdab.terminology import entries

ROOT = Path(__file__).resolve().parent
REQUESTS_DIR = ROOT / "requests"
EURLEX = "https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng"
ARTICLES = {
    "53": "Purposes for which health data can be processed for secondary use",
    "54": "Prohibited secondary use",
    "66": "Data minimisation and purpose limitation",
    "67": "Health data access applications",
    "68": "Data permits",
    "69": "Health data requests (anonymised statistics)",
    "71": "Right to opt out",
    "73": "Secure processing environment",
}
DECISION_STYLE = {
    "REJECT": ("error", "Rejected"),
    "APPROVE_STATISTICAL": ("success", "Approved: anonymised statistics"),
    "APPROVE_SPE": ("info", "Approved: pseudonymised data in a secure processing environment"),
}
PROTOTYPE_DISCLAIMER = (
    "Prototype: decisions are generated automatically for demonstration and are not legally binding "
    "decisions of a Health Data Access Body."
)
REQUEST_TYPES = {
    "Statistics only (health data request)": "health data request",
    "Access to pseudonymised records in a secure processing environment "
    "(health data access application)": "health data access application",
}

st.set_page_config(page_title="EHDS Health Data Access Body", layout="wide")


@st.cache_resource
def warm_data():
    return load_data()


def example_requests():
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(REQUESTS_DIR.glob("R*.json"))}


DIM_LABELS = {"region": "Region", "age_band": "Age band", "data_holder": "Hospital", "sex": "Sex"}


def answer_to_frame(answer, dims=()):
    """Flatten {group: {metric: value}} or deeper nesting into a table."""
    rows = []
    names = [DIM_LABELS.get(d.get("dimension"), d.get("dimension")) for d in dims]

    def col(i):
        return names[i] if i < len(names) else f"group_{i + 1}"

    def walk(node, path):
        if isinstance(node, dict) and node and all(not isinstance(v, dict) for v in node.values()):
            rows.append({**{col(i): g for i, g in enumerate(path)}, **node})
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])
        else:
            rows.append({**{col(i): g for i, g in enumerate(path)}, "value": node})

    walk(answer, [])
    df = pd.DataFrame(rows)
    return df.astype(str) if not df.empty else df


def store(resp):
    st.session_state.setdefault("decisions", {})[resp["request_id"]] = resp


def show_articles(articles):
    for a in articles or []:
        st.markdown(f"- **Art. {a}** – {ARTICLES.get(str(a), '')} ([Regulation (EU) 2025/327]({EURLEX}))")


def show_response(resp, staff=False):
    kind, title = DECISION_STYLE.get(resp["decision"], ("warning", resp["decision"]))
    getattr(st, kind)(f"**{title}** (`{resp['decision']}`)")
    st.markdown(f"**Reason:** {resp['reason']}")
    st.markdown("**Legal basis:**")
    show_articles(resp.get("articles"))
    if resp.get("needs_staff_review"):
        st.warning("Parts of this request could not be interpreted automatically. A staff member will review it.")
    answer = resp.get("answer")
    if resp["decision"] == "APPROVE_STATISTICAL" and answer:
        st.markdown("**Result (anonymised statistics)**")
        st.dataframe(answer_to_frame(answer, (resp.get("query_plan") or {}).get("group_by", [])), hide_index=True, use_container_width=True)
        st.caption("Small-cell rule: counts below 10 are shown as `<10`; percentages and means based on fewer "
                   "than 10 patients are `suppressed`. Patients who opted out (Art. 71) are excluded. "
                   "Values are rounded to one decimal, halves up.")
        st.download_button("Download statistics (JSON)", json.dumps(answer, indent=2, ensure_ascii=False),
                           file_name=f"{resp['request_id']}_statistics.json", mime="application/json")
    elif resp["decision"] == "APPROVE_SPE":
        spe = resp.get("spe") or {}
        st.markdown(f"**Cohort made available in the secure processing environment:** "
                    f"{(answer or {}).get('cohort_size', spe.get('cohort_size'))} patients")
        st.caption("The data can only be analysed inside the secure processing environment. Downloads are blocked "
                   "(Art. 73). Only aggregated results may leave it after review.")
    for w in resp.get("warnings") or []:
        st.caption(f"Note: {w}")
    st.caption(f"Processed in {resp.get('processing_time_s', 0):.3f} s")
    if staff:
        with st.expander("Staff details: triggered rules, generated query plan, privacy check"):
            st.markdown(f"**Purpose category:** {resp.get('purpose_category')}")
            st.json({k: resp.get(k) for k in ("triggered_rules", "privacy_check", "spe") if k in resp})
            st.markdown("**Query plan generated from the request's definitions**")
            st.json(resp.get("query_plan") or {})


_DISPLAY_TERMS = {
    "copd": "COPD",
    "egfr": "eGFR",
    "hba1c": "HbA1c",
}


def _display_term(term):
    return _DISPLAY_TERMS.get(term.casefold(), term[:1].upper() + term[1:])


def _builder_concepts(data, kind, table=None):
    table = table or {
        "condition": "conditions",
        "medication": "medications",
        "observation": "observations",
    }[kind]
    code_column = {
        "condition": "icd10" if table == "conditions" else "primary_icd10",
        "medication": "atc",
        "observation": "loinc",
    }[kind]
    system = {"condition": "ICD-10", "medication": "ATC", "observation": "LOINC"}[kind]
    available = data[table][code_column].astype(str)
    canonical = {}
    for _, concept in entries():
        if concept.kind == kind:
            canonical.setdefault(concept.term, concept)
    options = []
    for term, concept in sorted(canonical.items()):
        codes = list(concept.codes)
        present = any(
            available.eq(code).any()
            if kind == "observation"
            else available.str.startswith(code).any()
            for code in codes
        )
        if not present:
            continue
        display = _display_term(term)
        label = f"{display} ({system} {', '.join(codes)})"
        unit = ""
        if kind == "observation":
            rows = data[table].loc[data[table]["loinc"].isin(codes)]
            units = rows["unit"].dropna().astype(str).unique().tolist()
            unit = units[0] if units else ""
        options.append({
            "term": display,
            "canonical": term,
            "codes": codes,
            "system": system,
            "label": label,
            "unit": unit,
        })
    return options


def _year_bounds(from_year, to_year):
    if from_year is not None and to_year is not None and from_year > to_year:
        return None, None
    start = f"{from_year}-01-01" if from_year is not None else None
    end = f"{to_year}-12-31" if to_year is not None else None
    return start, end


def _age_bands(preset, custom):
    if preset == "10-year bands":
        return [
            {"lo": low, "hi": low + 9, "label": f"{low}-{low + 9}"}
            for low in range(0, 90, 10)
        ] + [{"lo": 90, "hi": None, "label": "90+"}]
    if preset == "18–39 / 40–64 / 65+":
        return [
            {"lo": 18, "hi": 39, "label": "18-39"},
            {"lo": 40, "hi": 64, "label": "40-64"},
            {"lo": 65, "hi": None, "label": "65+"},
        ]
    bands = []
    for token in custom.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", token)
        open_ended = re.fullmatch(r"\s*(\d+)\s*\+\s*", token)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                return []
            bands.append({"lo": low, "hi": high, "label": f"{low}-{high}"})
        elif open_ended:
            low = int(open_ended.group(1))
            bands.append({"lo": low, "hi": None, "label": f"{low}+"})
        elif token.strip():
            return []
    return bands


def _builder_code_rows(items):
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "items" in item:
            rows.extend(_builder_code_rows(item.get("items")))
            continue
        kind = item.get("type")
        if kind == "condition":
            system, codes = "ICD-10", item.get("icd10", [])
        elif kind == "medication":
            system, codes = "ATC", item.get("atc", [])
        elif kind == "lab":
            system, codes = "LOINC", [item.get("loinc", "")]
        elif kind == "hospital_stay" and item.get("icd10"):
            system, codes = "ICD-10", item.get("icd10", [])
        else:
            continue
        rows.append({
            "Concept": item.get("label", kind),
            "System": system,
            "Codes": ", ".join(str(code) for code in codes if code),
        })
    return rows


def _builder_criterion_row(row, data, period_start, period_end):
    row_id = row["id"]
    category = row["category"]
    criterion = None
    st.markdown(f"**{category}**")
    control_col, remove_col = st.columns([5, 1])
    exclude = control_col.toggle(
        "Exclude this criterion",
        key=f"builder_{row_id}_exclude",
    )
    if remove_col.button("Remove", key=f"builder_{row_id}_remove"):
        st.session_state["builder_criteria"] = [
            item for item in st.session_state["builder_criteria"]
            if item["id"] != row_id
        ]
        st.rerun()

    if category == "Diagnosis":
        options = _builder_concepts(data, "condition")
        if not options:
            st.caption("No dictionary diagnoses are present in the diagnosis list.")
            return None
        selected = st.selectbox(
            "Diagnosis",
            [option["label"] for option in options],
            key=f"builder_{row_id}_diagnosis",
        )
        concept = next(option for option in options if option["label"] == selected)
        timing_options = ["Diagnosed by 31 Dec 2025 (any time)"]
        if period_start and period_end:
            timing_options.append("Newly diagnosed within the time period")
        if period_end:
            timing_options.append("Diagnosed by the end of the time period")
        timing = st.radio(
            "Diagnosis timing",
            timing_options,
            key=f"builder_{row_id}_diagnosis_timing",
        )
        criterion = {
            "type": "condition",
            "label": concept["term"],
            "icd10": concept["codes"],
        }
        if timing == "Newly diagnosed within the time period":
            criterion.update({"onset_start": period_start, "onset_end": period_end})
        elif timing == "Diagnosed by the end of the time period":
            criterion["onset_end"] = period_end
        if not period_end:
            st.caption("Set a time period above to enable diagnosis timing filters.")
    elif category == "Medication":
        options = _builder_concepts(data, "medication")
        if not options:
            st.caption("No dictionary medications are present in the medication records.")
            return None
        selected = st.selectbox(
            "Medication",
            [option["label"] for option in options],
            key=f"builder_{row_id}_medication",
        )
        concept = next(option for option in options if option["label"] == selected)
        status_options = [
            "Active on 31 Dec 2025 (default)",
            "Ever prescribed",
        ]
        if period_start and period_end:
            status_options.append("Prescribed during the time period")
        status_label = st.radio(
            "Medication status",
            status_options,
            key=f"builder_{row_id}_medication_status",
        )
        status = {
            "Active on 31 Dec 2025 (default)": "active",
            "Ever prescribed": "ever",
            "Prescribed during the time period": "during_period",
        }[status_label]
        criterion = {
            "type": "medication",
            "label": concept["term"],
            "atc": concept["codes"],
            "status": status,
        }
        if status == "during_period":
            criterion.update({"period_start": period_start, "period_end": period_end})
        if not period_start or not period_end:
            st.caption("Set both years above to enable prescriptions during the time period.")
    elif category == "Laboratory measurement":
        options = _builder_concepts(data, "observation")
        if not options:
            st.caption("No dictionary laboratory measurements are present in the observations.")
            return None
        selected = st.selectbox(
            "Laboratory measurement",
            [option["label"] for option in options],
            key=f"builder_{row_id}_lab",
        )
        concept = next(option for option in options if option["label"] == selected)
        operator_labels = {
            ">": "Greater than (>)",
            ">=": "At least (≥)",
            "<": "Less than (<)",
            "<=": "At most (≤)",
            "present": "Has a measurement",
        }
        operator_label = st.selectbox(
            "Compare latest value",
            list(operator_labels.values()),
            key=f"builder_{row_id}_lab_operator",
        )
        operator = next(key for key, value in operator_labels.items() if value == operator_label)
        criterion = {
            "type": "lab",
            "label": concept["term"],
            "loinc": concept["codes"][0],
            "op": operator,
            "unit": concept["unit"],
        }
        if operator != "present":
            loinc = concept["codes"][0]
            threshold_defaults = {
                "4548-4": 8.0,
                "39156-5": 30.0,
                "8480-6": 140.0,
                "62238-1": 60.0,
            }
            criterion["value"] = st.number_input(
                f"Threshold ({concept['unit']})",
                value=threshold_defaults.get(loinc, 0.0),
                step=0.1,
                key=f"builder_{row_id}_lab_value_{loinc}",
            )
        if period_start:
            criterion["period_start"] = period_start
        if period_end:
            criterion["period_end"] = period_end
        st.caption(
            "Uses each patient's latest value within the time period "
            "(latest overall if no period)."
        )
    elif category == "Age":
        minimum_col, maximum_col = st.columns(2)
        has_minimum = minimum_col.checkbox(
            "Set minimum age",
            key=f"builder_{row_id}_age_min_enabled",
        )
        minimum = minimum_col.number_input(
            "Minimum age",
            min_value=0,
            max_value=120,
            value=18,
            disabled=not has_minimum,
            key=f"builder_{row_id}_age_min",
        )
        has_maximum = maximum_col.checkbox(
            "Set maximum age",
            key=f"builder_{row_id}_age_max_enabled",
        )
        maximum = maximum_col.number_input(
            "Maximum age",
            min_value=0,
            max_value=120,
            value=65,
            disabled=not has_maximum,
            key=f"builder_{row_id}_age_max",
        )
        criterion = {"type": "age"}
        if has_minimum:
            criterion["min"] = int(minimum)
        if has_maximum:
            criterion["max"] = int(maximum)
        if not has_minimum and not has_maximum:
            st.caption("Set a minimum or maximum age.")
            return None
        criterion["label"] = "Age"
    elif category == "Sex":
        value = st.selectbox(
            "Sex",
            ["female", "male"],
            key=f"builder_{row_id}_sex",
        )
        criterion = {"type": "sex", "label": value, "value": value}
    elif category in {"Region", "Hospital (data holder)"}:
        kind = "region" if category == "Region" else "data_holder"
        values = data["categories"][kind]
        selected = st.multiselect(
            category,
            values,
            key=f"builder_{row_id}_{kind}",
        )
        if not selected:
            st.caption(f"Select one or more {category.casefold()} values.")
            return None
        criterion = {
            "type": kind,
            "label": category,
            "values": selected,
        }
    elif category == "Hospital stay (main diagnosis)":
        options = _builder_concepts(data, "condition", "encounters")
        options = [{
            "term": "Any main diagnosis",
            "codes": [],
            "label": "Any main diagnosis",
        }, *options]
        selected = st.selectbox(
            "Inpatient diagnosis",
            [option["label"] for option in options],
            key=f"builder_{row_id}_hospital_stay",
        )
        concept = next(option for option in options if option["label"] == selected)
        criterion = {
            "type": "hospital_stay",
            "label": concept["term"],
            "icd10": concept["codes"],
        }
        if period_start:
            criterion["period_start"] = period_start
        if period_end:
            criterion["period_end"] = period_end
    if criterion is not None:
        criterion["exclude"] = exclude
    return criterion


def page_applicant():
    st.header("Applicant portal")
    st.caption(PROTOTYPE_DISCLAIMER)
    examples = example_requests()
    describe_tab, builder_tab = st.tabs(["Describe your research", "Build your research cohort"])
    with describe_tab:
        st.write("Definitions are optional; the built-in dictionary is used otherwise.")
        with st.form("applicant_request_form"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Applicant name")
            org = c2.text_input("Organisation")
            country = c3.text_input("Country", "AT")
            request_type = st.selectbox("Type of request", list(REQUEST_TYPES))
            purpose = st.text_area("Purpose")
            question = st.text_area("Question")
            definitions = st.text_area(
                "Definitions (one per line)",
                height=150,
                help="Optional. If omitted, the built-in terminology dictionary is used.",
            )
            submit_form = st.form_submit_button("Submit request", type="primary")
        if submit_form:
            requested_as = REQUEST_TYPES[request_type]
            request = {
                "id": st.session_state.get("form_id", "NEW-1"),
                "applicant": {"name": name, "organisation": org, "country": country},
                "requested_as": requested_as,
                "requested_format": (
                    "anonymised statistics"
                    if requested_as == "health data request"
                    else "pseudonymised records"
                ),
                "purpose": purpose,
                "question": question,
                "definitions": [d.strip() for d in definitions.splitlines() if d.strip()],
            }
            with st.spinner("Assessing request..."):
                resp = process_request(request)
            store(resp)
            st.session_state["last"] = resp["request_id"]

    with builder_tab:
        data = warm_data()
        st.write("Choose a population, add optional criteria, and preview the request before submission.")
        applicant_col1, applicant_col2, applicant_col3 = st.columns(3)
        builder_name = applicant_col1.text_input("Applicant name", key="builder_applicant_name")
        builder_org = applicant_col2.text_input("Organisation", key="builder_applicant_org")
        builder_country = applicant_col3.text_input(
            "Country",
            "AT",
            key="builder_applicant_country",
        )
        builder_purpose = st.text_area(
            "Purpose *",
            help="Checked against EHDS Art. 53/54",
            key="builder_purpose",
        )
        builder_type = st.selectbox(
            "Type of request",
            list(REQUEST_TYPES),
            key="builder_request_type",
        )
        builder_requested_as = REQUEST_TYPES[builder_type]
        record_types = []
        if builder_requested_as == "health data access application":
            record_types = st.multiselect(
                "Records to access in the SPE",
                ["conditions", "medications", "observations", "encounters"],
                help="The purpose must explain why anonymised statistics are not enough.",
                key="builder_record_types",
            )

        year_columns = (
            ("conditions", "onset_date"),
            ("observations", "date"),
            ("medications", "start_date"),
            ("encounters", "start_date"),
        )
        years_by_table = {}
        years = set()
        for table, column in year_columns:
            table_years = {
                int(value)
                for value in data[table][column].dropna().dt.year.unique()
                if int(value) <= 2025
            }
            years_by_table[table] = table_years
            years.update(table_years)
        year_options = [None, *sorted(years)]
        record_year_sets = [
            years_by_table[table]
            for table in ("observations", "medications", "encounters")
        ]
        if all(record_year_sets):
            shared_years = set.intersection(*record_year_sets)
            if shared_years:
                coverage = f"{min(shared_years)}–{max(shared_years)}"
            else:
                coverage = "not shared across all three record types"
        else:
            coverage = "not available"
        year_col1, year_col2 = st.columns(2)
        from_year = year_col1.selectbox(
            "From year",
            year_options,
            format_func=lambda value: "No start year" if value is None else str(value),
            key="builder_from_year",
        )
        to_year = year_col2.selectbox(
            "To year",
            year_options,
            format_func=lambda value: "No end year" if value is None else str(value),
            key="builder_to_year",
        )
        st.caption(f"Lab, medication and hospital-stay records cover {coverage}")
        period_start, period_end = _year_bounds(from_year, to_year)
        invalid_period = from_year is not None and to_year is not None and from_year > to_year
        if invalid_period:
            st.error("The start year must not be after the end year.")

        st.radio(
            "Patients must meet:",
            ["ALL of the criteria", "ANY of the criteria"],
            horizontal=True,
            key="builder_match",
        )
        categories = [
            "Diagnosis",
            "Medication",
            "Laboratory measurement",
            "Age",
            "Sex",
            "Region",
            "Hospital (data holder)",
            "Hospital stay (main diagnosis)",
        ]
        category = st.selectbox("Criterion type", categories, key="builder_new_category")
        if st.button("Add criterion", key="builder_add_criterion"):
            row_id = st.session_state.get("builder_next_criterion_id", 0) + 1
            st.session_state["builder_next_criterion_id"] = row_id
            st.session_state.setdefault("builder_criteria", []).append({
                "id": row_id,
                "category": category,
            })
            st.rerun()

        st.session_state.setdefault("builder_criteria", [])
        criterion_items = []
        for row in list(st.session_state["builder_criteria"]):
            with st.container(border=True):
                criterion = _builder_criterion_row(row, data, period_start, period_end)
            if criterion is not None:
                criterion_items.append(criterion)
        match = "any" if st.session_state.get("builder_match") == "ANY of the criteria" else "all"

        group_by_choices = {
            "Sex": "sex",
            "Age band": "age_band",
            "Region": "region",
            "Hospital": "data_holder",
        }
        selected_groups = st.multiselect(
            "Group results by",
            list(group_by_choices),
            key="builder_group_by",
        )
        group_by = []
        for label in selected_groups:
            dimension = group_by_choices[label]
            if dimension == "age_band":
                preset = st.selectbox(
                    "Age band preset",
                    ["10-year bands", "18–39 / 40–64 / 65+", "Custom"],
                    key="builder_age_band_preset",
                )
                custom_bands = ""
                if preset == "Custom":
                    custom_bands = st.text_input(
                        "Custom age bands",
                        "40-49, 50-59, 60+",
                        key="builder_custom_age_bands",
                    )
                bands = _age_bands(preset, custom_bands)
                if bands:
                    group_by.append({"dimension": "age_band", "bands": bands})
                else:
                    st.warning("Enter valid age bands such as 40-49, 50-59, 60+.")
            else:
                group_by.append({"dimension": dimension})

        metrics = [{"type": "count_patients", "label": "patients"}]
        if st.checkbox("Mean of latest lab value", key="builder_mean_enabled"):
            lab_options = _builder_concepts(data, "observation")
            if lab_options:
                selected_mean_lab = st.selectbox(
                    "Laboratory value to average",
                    [option["label"] for option in lab_options],
                    key="builder_mean_lab",
                )
                mean_lab = next(
                    option for option in lab_options if option["label"] == selected_mean_lab
                )
                metric_label = f"mean latest {mean_lab['term']} ({mean_lab['unit']})"
                mean_metric = {
                    "type": "mean_latest_lab",
                    "label": metric_label,
                    "loinc": mean_lab["codes"][0],
                }
                if period_start:
                    mean_metric["period_start"] = period_start
                if period_end:
                    mean_metric["period_end"] = period_end
                metrics.append(mean_metric)
            else:
                st.caption("No dictionary laboratory measurements are available.")

        criteria = {
            "population": {"match": match, "items": criterion_items},
            "group_by": group_by,
            "metrics": metrics,
            "record_types": record_types,
        }
        request_number = st.session_state.get("builder_request_number", 0) + 1
        requested_format = (
            "anonymised statistics"
            if builder_requested_as == "health data request"
            else "pseudonymised records"
        )
        builder_request = {
            "id": f"BUILDER-{request_number}",
            "applicant": {
                "name": builder_name,
                "organisation": builder_org,
                "country": builder_country,
            },
            "requested_as": builder_requested_as,
            "requested_format": requested_format,
            "purpose": builder_purpose,
            "question": describe_criteria(criteria),
            "definitions": criteria_to_definitions(criteria),
            "criteria": criteria,
        }
        st.info(describe_criteria(criteria))
        code_rows = _builder_code_rows(criterion_items)
        if code_rows:
            st.table(pd.DataFrame(code_rows))
        with st.expander("Request JSON"):
            st.json(builder_request)
        if st.button(
            "Submit request",
            key="builder_submit",
            type="primary",
            disabled=not builder_purpose.strip() or not criterion_items or invalid_period,
        ):
            with st.spinner("Assessing request..."):
                resp = process_request(builder_request)
            store(resp)
            st.session_state["last"] = resp["request_id"]
            st.session_state["builder_request_number"] = request_number

    with st.expander("Advanced options", expanded=False):
        source = st.radio(
            "Request source",
            ["Example request", "Upload JSON", "Paste JSON"],
            horizontal=True,
        )
        request = None
        if source == "Example request":
            rid = st.selectbox("Example", list(examples))
            request = examples[rid]
            a = request.get("applicant", {})
            st.markdown(f"**Applicant:** {a.get('name')}, {a.get('organisation')} ({a.get('country')})  \n"
                        f"**Requested as:** {request.get('requested_as')}  \n"
                        f"**Purpose:** {request.get('purpose')}  \n**Question:** {request.get('question')}")
            with st.expander("Full request JSON"):
                st.json(request)
        elif source == "Upload JSON":
            f = st.file_uploader("Request JSON", type="json")
            if f:
                try:
                    request = json.loads(f.read().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    st.error(f"Invalid JSON file: {e}")
        elif source == "Paste JSON":
            txt = st.text_area("Request JSON", height=300)
            if txt.strip():
                try:
                    request = json.loads(txt)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        if request and st.button("Submit this request", type="primary"):
            with st.spinner("Assessing request..."):
                resp = process_request(request)
            store(resp)
            st.session_state["last"] = resp["request_id"]
    last = st.session_state.get("decisions", {}).get(st.session_state.get("last"))
    if last:
        st.subheader(f"Decision for {last['request_id']}")
        show_response(last)


def page_staff():
    st.header("Staff dashboard")
    if st.button("Process all example requests"):
        for req in example_requests().values():
            store(process_request(req))
    decisions = st.session_state.get("decisions", {})
    if not decisions:
        st.info("No requests processed yet in this session.")
    else:
        overview = pd.DataFrame([{
            "request": r["request_id"], "decision": r["decision"],
            "articles": ", ".join(r.get("articles") or []),
            "purpose category": r.get("purpose_category"),
            "needs review": r.get("needs_staff_review"),
            "privacy check": "passed" if not (r.get("privacy_check") or {}).get("violations") else "FAILED",
            "time (s)": round(r.get("processing_time_s", 0), 3),
        } for r in decisions.values()])
        st.dataframe(overview, hide_index=True, use_container_width=True)
        rid = st.selectbox("Inspect request", list(decisions))
        show_response(decisions[rid], staff=True)
    st.subheader("Audit log")
    log = Path(config.AUDIT_LOG)
    if log.exists():
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(lines[::-1]).astype(str), hide_index=True, use_container_width=True)
    else:
        st.caption("Empty.")


def page_spe():
    st.header("Secure processing environment (SPE)")
    st.write("Approved applicants analyse pseudonymised records here. Data cannot be downloaded or exported.")
    root = Path(config.SPE_ROOT)
    workspaces = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not workspaces:
        st.info("No SPE workspaces yet. Submit an access application that is approved for an SPE (e.g. R05).")
        return
    ws = st.selectbox("Workspace", workspaces)
    env = SecureProcessingEnvironment(root / ws)
    manifest = json.loads((root / ws / "manifest.json").read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients in cohort", manifest.get("cohort_size"))
    c2.metric("Legal basis", "Art. " + ", ".join(manifest.get("articles", [])))
    c3.metric("Downloads", "Blocked" if not manifest.get("download_allowed") else "Allowed")
    with st.expander("Workspace manifest"):
        st.json(manifest)
    table = st.selectbox("Table", env.list_tables())
    df = env.read_table(table)
    st.markdown(f"**{table}**: {len(df)} rows, {df['patient_id'].nunique() if 'patient_id' in df else '-'} patients")
    # st.table has no export toolbar, unlike st.dataframe, so records cannot be saved from the browser view.
    st.table(df.head(50))
    st.caption("Showing the first 50 rows. Viewing is permitted inside the SPE; exporting is not.")
    with st.expander("Analyse inside the SPE"):
        col = st.selectbox("Column", list(df.columns))
        st.table(df[col].value_counts().head(20).rename("rows").reset_index().astype(str))
    if st.button("Download this table", type="secondary"):
        try:
            env.request_download(table, user="applicant")
        except DownloadBlockedError as e:
            st.error(f"Download blocked: {e}")


def page_evaluation():
    st.header("Evaluation on the 10 challenge requests")
    st.write("Runs every request 3 times and compares with `requests/expected.json`.")
    if st.button("Run evaluation", type="primary"):
        import subprocess
        import sys
        with st.spinner("Running..."):
            out = subprocess.run([sys.executable, str(ROOT / "evaluate.py")], cwd=ROOT,
                                 capture_output=True, text=True)
        st.code(out.stdout + out.stderr)


warm_data()
PAGES = {"Applicant portal": page_applicant, "Staff dashboard": page_staff,
         "Secure processing environment": page_spe, "Evaluation": page_evaluation}
st.sidebar.title("EHDS Health Data Access Body")
st.sidebar.caption("Simulated HDAB for AIM Hackathon Vienna 2026, Track 3. Synthetic data only.")
st.sidebar.caption(PROTOTYPE_DISCLAIMER)
PAGES[st.sidebar.radio("View", list(PAGES))]()
