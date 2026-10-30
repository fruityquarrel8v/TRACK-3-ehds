"""Streamlit web interface for the simulated EHDS Health Data Access Body.

Run: streamlit run app.py
Patient-level data never leaves this machine; no external services are called.
"""
import copy
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from ehds_hdab import config
from ehds_hdab.criteria import criteria_to_definitions, describe_criteria
from ehds_hdab.data import load_data
from ehds_hdab.nl import interpret_question
from ehds_hdab.pipeline import process_request
from ehds_hdab.spe import DownloadBlockedError, SecureProcessingEnvironment
from ehds_hdab.viz import answer_table, interpret_chart_command, methodology_text, render_chart
from ehds_hdab.terminology import display_name, entries

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


def store(resp, request=None):
    st.session_state.setdefault("decisions", {})[resp["request_id"]] = resp
    if request is not None:
        st.session_state.setdefault("requests", {})[resp["request_id"]] = request


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
    if resp.get("withheld"):
        reason = next(
            (
                warning for warning in resp.get("warnings", [])
                if warning.startswith("Results withheld:")
            ),
            "Results withheld for differencing protection. Staff review required.",
        )
        st.warning(reason)
    elif resp["decision"] == "APPROVE_STATISTICAL" and answer:
        st.markdown("**Result (anonymised statistics)**")
        st.dataframe(answer_to_frame(answer, (resp.get("query_plan") or {}).get("group_by", [])), hide_index=True, use_container_width=True)
        st.caption("Small-cell rule: counts below 10 are shown as `<10`; percentages and means based on fewer "
                   "than 10 patients are `suppressed`. Patients who opted out (Art. 71) are excluded. "
                   "Values are rounded to one decimal, halves up.")
        st.download_button("Download statistics (JSON)", json.dumps(answer, indent=2, ensure_ascii=False),
                           file_name=f"{resp['request_id']}_statistics.json", mime="application/json")
        if not staff:
            _chart_section(resp)
    elif resp["decision"] == "APPROVE_SPE":
        spe = resp.get("spe") or {}
        st.markdown(f"**Cohort made available in the secure processing environment:** "
                    f"{(answer or {}).get('cohort_size', spe.get('cohort_size'))} patients")
        st.caption("The data can only be analysed inside the secure processing environment. Downloads are blocked "
                   "(Art. 73). Only aggregated results may leave it after review.")
    for w in resp.get("warnings") or []:
        if resp.get("withheld") and w.startswith("Results withheld:"):
            continue
        st.caption(f"Note: {w}")
    st.caption(f"Processed in {resp.get('processing_time_s', 0):.3f} s")
    if staff:
        with st.expander("Staff details: triggered rules, generated query plan, privacy check"):
            st.markdown(f"**Purpose category:** {resp.get('purpose_category')}")
            st.json({k: resp.get(k) for k in ("triggered_rules", "privacy_check", "spe") if k in resp})
            st.markdown("**Query plan generated from the request's definitions**")
            st.json(resp.get("query_plan") or {})


def _regroup(resp, result):
    request = st.session_state.get("requests", {}).get(resp["request_id"])
    if not request or not isinstance(request.get("criteria"), dict):
        st.info("Regrouping is available for requests made with the cohort builder or the research question box.")
        return
    regrouped = copy.deepcopy(request)
    regrouped["criteria"]["group_by"] = [
        {"dimension": "age_band", "bands": _age_bands("10-year bands", "")} if dimension == "age_band"
        else {"dimension": dimension}
        for dimension in result["regroup"]
    ]
    number = st.session_state.get("regroup_number", 0) + 1
    st.session_state["regroup_number"] = number
    regrouped["id"] = f"{str(resp['request_id']).split('-G')[0]}-G{number}"
    with st.spinner("Re-running the request through the approval process..."):
        new_resp = process_request(regrouped)
    store(new_resp, regrouped)
    st.session_state["last"] = new_resp["request_id"]
    st.session_state.setdefault("chart_specs", {})[new_resp["request_id"]] = result["spec"]
    st.rerun()


def _chart_section(resp):
    rid = resp["request_id"]
    plan = resp.get("query_plan") or {}
    group_by = plan.get("group_by", [])
    table = answer_table(resp["answer"], group_by)
    dims = [DIM_LABELS.get(d.get("dimension"), d.get("dimension")) for d in group_by]
    metrics = [c for c in table.columns if c not in dims]
    if table.empty or not metrics:
        return
    st.markdown("**Visualise these results**")
    st.caption("Charts are drawn only from the anonymised table above, so they are privacy-safe to export.")
    specs = st.session_state.setdefault("chart_specs", {})
    spec = specs.get(rid) or {"type": "bar", "metric": metrics[0], "x": dims[0] if dims else None, "color": None}
    examples = [
        f"Bar chart of {metrics[0]} by {dims[0]}" if dims else f"Bar chart of {metrics[0]}",
        "Change this into a line graph",
        "Group the results by sex and age",
    ]
    command = st.text_input("Describe the chart you want", key=f"chart_cmd_{rid}", placeholder=examples[0])
    chosen = None
    for i, (col, example) in enumerate(zip(st.columns(3), examples)):
        if col.button(example, key=f"chart_ex_{rid}_{i}"):
            chosen = example
    command = chosen or command
    message = ""
    if command:
        result = interpret_chart_command(command, spec, metrics, dims)
        if result["regroup"]:
            _regroup(resp, result)
            return
        if result["ok"]:
            spec = result["spec"]
            specs[rid] = spec
        else:
            message = result["message"]
    if message:
        st.info(message)
    request = st.session_state.get("requests", {}).get(rid) or {}
    criteria = request.get("criteria") if isinstance(request.get("criteria"), dict) else None
    population_text = describe_criteria(criteria) if criteria else str(request.get("question") or rid)
    period_text = _criteria_period_summary(criteria) if criteria else "As defined in the request"
    title = f"{spec['metric']} by {spec['x']}" if spec.get("x") else str(spec["metric"])
    if spec.get("color"):
        title += f" and {spec['color']}"
    fig, png = render_chart(table, spec, title, population_text, period_text, "")
    st.pyplot(fig)
    with st.expander("Statistical methodology and definitions"):
        st.write(methodology_text(spec, resp))
        st.write(f"**Population:** {population_text}")
        st.write(f"**Time period:** {period_text}")
    st.download_button("Download chart (PNG)", png, file_name=f"{rid}_chart.png", mime="image/png",
                       key=f"chart_png_{rid}")


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
        display = display_name(term)
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


def _criterion_leaves(items):
    leaves = []
    has_group = False
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "items" in item:
            has_group = True
            nested, _ = _criterion_leaves(item.get("items"))
            leaves.extend(nested)
        else:
            leaves.append(item)
    return leaves, has_group


def _matching_builder_option(options, codes, label=None):
    codes = {str(code) for code in codes or []}
    if codes:
        for option in options:
            if codes.intersection(str(code) for code in option.get("codes", [])):
                return option
    if label:
        lowered = str(label).casefold()
        for option in options:
            if lowered in {
                str(option.get("label", "")).casefold(),
                str(option.get("term", "")).casefold(),
                str(option.get("canonical", "")).casefold(),
            }:
                return option
    return options[0] if options else None


def _prefill_builder(criteria, applicant, purpose, request_type):
    data = warm_data()
    population = criteria.get("population", {})
    items = population.get("items", []) if isinstance(population, dict) else []
    leaves, has_nested_group = _criterion_leaves(items)
    flat_any = (
        isinstance(population, dict)
        and str(population.get("match", "all")).casefold() == "any"
    )
    st.session_state["builder_match"] = (
        "ANY of the criteria" if flat_any else "ALL of the criteria"
    )
    st.session_state["builder_criteria"] = []
    st.session_state["builder_applicant_name"] = applicant.get("name", "")
    st.session_state["builder_applicant_org"] = applicant.get("organisation", "")
    st.session_state["builder_applicant_country"] = applicant.get("country", "AT")
    st.session_state["builder_purpose"] = purpose
    st.session_state["builder_request_type"] = request_type
    st.session_state["builder_record_types"] = list(criteria.get("record_types", []))

    starts = []
    ends = []
    for item in leaves:
        for key in ("period_start", "onset_start"):
            if item.get(key):
                starts.append(int(str(item[key])[:4]))
        for key in ("period_end", "onset_end"):
            if item.get(key):
                ends.append(int(str(item[key])[:4]))
    for metric in criteria.get("metrics", []):
        if metric.get("period_start"):
            starts.append(int(str(metric["period_start"])[:4]))
        if metric.get("period_end"):
            ends.append(int(str(metric["period_end"])[:4]))
    st.session_state["builder_from_year"] = min(starts) if starts else None
    st.session_state["builder_to_year"] = max(ends) if ends else None

    dimension_labels = {
        "sex": "Sex",
        "age_band": "Age band",
        "region": "Region",
        "data_holder": "Hospital",
    }
    group_by = criteria.get("group_by", [])
    st.session_state["builder_group_by"] = [
        dimension_labels[item["dimension"]]
        for item in group_by
        if item.get("dimension") in dimension_labels
    ]
    age_group = next(
        (item for item in group_by if item.get("dimension") == "age_band"),
        None,
    )
    if age_group:
        bands = age_group.get("bands", [])
        if bands == _age_bands("10-year bands", ""):
            st.session_state["builder_age_band_preset"] = "10-year bands"
        elif bands == _age_bands("18–39 / 40–64 / 65+", ""):
            st.session_state["builder_age_band_preset"] = "18–39 / 40–64 / 65+"
        else:
            st.session_state["builder_age_band_preset"] = "Custom"
            st.session_state["builder_custom_age_bands"] = ", ".join(
                f"{band.get('lo')}-{band.get('hi')}"
                if band.get("hi") is not None
                else f"{band.get('lo')}+"
                for band in bands
            )

    mean_metric = next(
        (item for item in criteria.get("metrics", [])
         if item.get("type") == "mean_latest_lab"),
        None,
    )
    st.session_state["builder_mean_enabled"] = mean_metric is not None
    if mean_metric:
        mean_option = _matching_builder_option(
            _builder_concepts(data, "observation"),
            [mean_metric.get("loinc")],
        )
        if mean_option:
            st.session_state["builder_mean_lab"] = mean_option["label"]

    categories = {
        "condition": ("Diagnosis", "condition", "icd10", "diagnosis"),
        "medication": ("Medication", "medication", "atc", "medication"),
        "lab": ("Laboratory measurement", "observation", "loinc", "lab"),
        "age": ("Age", None, None, "age"),
        "sex": ("Sex", None, None, "sex"),
        "region": ("Region", None, None, "region"),
        "data_holder": ("Hospital (data holder)", None, None, "data_holder"),
        "hospital_stay": (
            "Hospital stay (main diagnosis)",
            "condition",
            "icd10",
            "hospital_stay",
        ),
    }
    operator_labels = {
        ">": "Greater than (>)",
        ">=": "At least (≥)",
        "<": "Less than (<)",
        "<=": "At most (≤)",
        "present": "Has a measurement",
    }
    medication_status_labels = {
        "active": "Active on 31 Dec 2025 (default)",
        "ever": "Ever prescribed",
        "during_period": "Prescribed during the time period",
    }

    for index, item in enumerate(leaves, start=1):
        kind = item.get("type")
        if kind not in categories:
            continue
        category, option_kind, code_field, key_suffix = categories[kind]
        row_id = index
        st.session_state["builder_criteria"].append(
            {"id": row_id, "category": category}
        )
        st.session_state["builder_next_criterion_id"] = row_id
        st.session_state[f"builder_{row_id}_exclude"] = bool(item.get("exclude"))
        if kind in {"condition", "medication", "lab", "hospital_stay"}:
            table = "encounters" if kind == "hospital_stay" else None
            options = _builder_concepts(data, option_kind, table)
            codes = item.get(code_field, [])
            if kind == "lab":
                codes = [item.get("loinc", "")]
            option = _matching_builder_option(options, codes, item.get("label"))
            if option:
                widget_key = {
                    "condition": "diagnosis",
                    "medication": "medication",
                    "lab": "lab",
                    "hospital_stay": "hospital_stay",
                }[kind]
                st.session_state[f"builder_{row_id}_{widget_key}"] = option["label"]
        if kind == "condition":
            if item.get("onset_start") and item.get("onset_end"):
                timing = "Newly diagnosed within the time period"
            elif item.get("onset_start"):
                timing = "Diagnosed since the start of the time period"
            elif item.get("onset_end"):
                timing = "Diagnosed by the end of the time period"
            else:
                timing = "Diagnosed by 31 Dec 2025 (any time)"
            st.session_state[f"builder_{row_id}_diagnosis_timing"] = timing
        elif kind == "medication":
            st.session_state[f"builder_{row_id}_medication_status"] = (
                medication_status_labels.get(
                    item.get("status"), "Active on 31 Dec 2025 (default)"
                )
            )
        elif kind == "lab":
            st.session_state[f"builder_{row_id}_lab_operator"] = operator_labels.get(
                item.get("op"), "Has a measurement"
            )
            if item.get("loinc") and item.get("op") != "present":
                st.session_state[
                    f"builder_{row_id}_lab_value_{item['loinc']}"
                ] = float(item.get("value", 0.0))
        elif kind == "age":
            st.session_state[f"builder_{row_id}_age_min_enabled"] = item.get("min") is not None
            st.session_state[f"builder_{row_id}_age_max_enabled"] = item.get("max") is not None
            if item.get("min") is not None:
                st.session_state[f"builder_{row_id}_age_min"] = int(item["min"])
            if item.get("max") is not None:
                st.session_state[f"builder_{row_id}_age_max"] = int(item["max"])
        elif kind == "sex":
            st.session_state[f"builder_{row_id}_sex"] = item.get("value", "female")
        elif kind == "region":
            st.session_state[f"builder_{row_id}_region"] = list(item.get("values", []))
        elif kind == "data_holder":
            st.session_state[f"builder_{row_id}_data_holder"] = list(item.get("values", []))

    return has_nested_group


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
        # Rerun only after every other row has rendered; a rerun here would drop their widget state.
        st.session_state["builder_rerun_needed"] = True
        return None

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
        elif period_start:
            timing_options.append("Diagnosed since the start of the time period")
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
        elif timing == "Diagnosed since the start of the time period":
            criterion["onset_start"] = period_start
        elif timing == "Diagnosed by the end of the time period":
            criterion["onset_end"] = period_end
        if not period_start and not period_end:
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


def _criteria_period_summary(criteria):
    population = criteria.get("population", {})
    items = population.get("items", []) if isinstance(population, dict) else []
    leaves, _ = _criterion_leaves(items)
    bounds = []
    for item in leaves:
        start = item.get("period_start") or item.get("onset_start")
        end = item.get("period_end") or item.get("onset_end")
        if start or end:
            bounds.append((start, end))
    for metric in criteria.get("metrics", []):
        start, end = metric.get("period_start"), metric.get("period_end")
        if start or end:
            bounds.append((start, end))
    if not bounds:
        return "No time period applied."
    labels = []
    for start, end in dict.fromkeys(bounds):
        if start and end:
            labels.append(f"{start} to {end}")
        elif start:
            labels.append(f"Since {start}")
        else:
            labels.append(f"Through {end}")
    return "; ".join(labels)


def page_applicant():
    st.header("Applicant portal")
    st.caption(PROTOTYPE_DISCLAIMER)
    examples = example_requests()
    describe_tab, builder_tab = st.tabs(["Describe your research", "Build your research cohort"])
    with describe_tab:
        st.write("Describe the research question you want answered.")
        applicant_col1, applicant_col2, applicant_col3 = st.columns(3)
        name = applicant_col1.text_input("Applicant name", key="nl_applicant_name")
        organisation = applicant_col2.text_input(
            "Organisation", key="nl_applicant_organisation"
        )
        country = applicant_col3.text_input("Country", "AT", key="nl_applicant_country")
        purpose = st.text_area(
            "Purpose *",
            help="Checked against EHDS Art. 53/54",
            key="nl_purpose",
        )
        request_type = st.selectbox(
            "Type of request",
            list(REQUEST_TYPES),
            key="nl_request_type",
        )
        requested_as = REQUEST_TYPES[request_type]
        record_types = []
        if requested_as == "health data access application":
            record_types = st.multiselect(
                "Records to access in the SPE",
                ["conditions", "medications", "observations", "encounters"],
                help="The purpose must explain why anonymised statistics are not enough.",
                key="nl_record_types",
            )
        question = st.text_area(
            "Research question",
            placeholder=(
                "Compare HbA1c levels between male and female patients with "
                "type 2 diabetes aged 40-65 during 2025."
            ),
            key="nl_question",
        )
        if st.button("Interpret my question", key="nl_interpret"):
            st.session_state["nl_interpretation"] = interpret_question(question)
            st.session_state["nl_interpreted_text"] = question

        interpretation = st.session_state.get("nl_interpretation")
        if interpretation is not None:
            criteria = interpretation.criteria
            criteria["record_types"] = record_types
            st.subheader("Interpretation")
            description = describe_criteria(criteria)
            metric_labels = [
                metric.get("label")
                for metric in criteria.get("metrics", [])
                if metric.get("label")
            ]
            if metric_labels:
                description += " Outputs: " + ", ".join(metric_labels) + "."
            st.info(description)
            code_rows = _builder_code_rows(
                criteria.get("population", {}).get("items", [])
            )
            if code_rows:
                st.table(pd.DataFrame(code_rows))

            group_labels = {
                "sex": "Sex",
                "age_band": "Age band",
                "region": "Region",
                "data_holder": "Hospital",
            }
            groups = [
                group_labels.get(item.get("dimension"), item.get("dimension"))
                for item in criteria.get("group_by", [])
            ]
            outputs = [
                item.get("label", item.get("type", "Output"))
                for item in criteria.get("metrics", [])
            ]
            st.markdown(f"**Time period:** {_criteria_period_summary(criteria)}")
            st.markdown(
                f"**Grouping:** {', '.join(groups) if groups else 'No grouping'}"
            )
            st.markdown(f"**Outputs:** {', '.join(outputs)}")
            st.markdown("**Assumptions and documented defaults**")
            if interpretation.assumptions:
                for assumption in interpretation.assumptions:
                    st.markdown(f"- {assumption}")
            else:
                st.write("No additional assumptions or defaults applied.")
            if interpretation.unrecognised:
                st.warning(
                    "Not recognised (not used): "
                    + ", ".join(interpretation.unrecognised)
                )
            if not interpretation.ok():
                st.error(
                    "Nothing clinical or demographic was recognised. Please "
                    "refine the question or use the cohort builder."
                )
            if st.session_state.get("nl_interpreted_text") != question:
                st.caption("Edit the question and interpret it again before submitting.")

            submit_col, correct_col = st.columns(2)
            if submit_col.button(
                "Confirm and submit",
                key="nl_submit",
                type="primary",
                disabled=(
                    not interpretation.ok()
                    or not purpose.strip()
                    or st.session_state.get("nl_interpreted_text") != question
                ),
            ):
                request_number = st.session_state.get("nl_request_number", 0) + 1
                request = {
                    "id": f"NL-{request_number}",
                    "applicant": {
                        "name": name,
                        "organisation": organisation,
                        "country": country,
                    },
                    "requested_as": requested_as,
                    "requested_format": (
                        "anonymised statistics"
                        if requested_as == "health data request"
                        else "pseudonymised records"
                    ),
                    "purpose": purpose,
                    "question": st.session_state.get("nl_interpreted_text", question),
                    "definitions": criteria_to_definitions(criteria),
                    "criteria": criteria,
                }
                with st.spinner("Assessing request..."):
                    resp = process_request(request)
                store(resp, request)
                st.session_state["last"] = resp["request_id"]
                st.session_state["nl_request_number"] = request_number
            if correct_col.button("Correct in cohort builder", key="nl_correct"):
                nested = _prefill_builder(
                    criteria,
                    {
                        "name": name,
                        "organisation": organisation,
                        "country": country,
                    },
                    purpose,
                    request_type,
                )
                st.success("Loaded into the cohort builder tab — open it to edit.")
                if nested:
                    st.caption(
                        "Nested groups cannot be edited in the cohort builder yet; "
                        "use Confirm and submit to preserve the interpretation."
                    )

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
            store(resp, builder_request)
            st.session_state["last"] = resp["request_id"]
            st.session_state["builder_request_number"] = request_number

    with st.expander("Advanced options", expanded=False):
        source = st.radio(
            "Request source",
            [
                "Example request",
                "Upload JSON",
                "Paste JSON",
                "Fill in a form (challenge format)",
            ],
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
        elif source == "Fill in a form (challenge format)":
            with st.form("advanced_challenge_form"):
                form_col1, form_col2, form_col3 = st.columns(3)
                form_name = form_col1.text_input("Applicant name")
                form_org = form_col2.text_input("Organisation")
                form_country = form_col3.text_input("Country", "AT")
                form_request_type = st.selectbox(
                    "Type of request", list(REQUEST_TYPES)
                )
                form_purpose = st.text_area("Purpose")
                form_question = st.text_area("Question")
                form_definitions = st.text_area(
                    "Definitions (one per line)",
                    height=150,
                    help="Optional. If omitted, the built-in terminology dictionary is used.",
                )
                submit_challenge = st.form_submit_button(
                    "Submit this request", type="primary"
                )
            if submit_challenge:
                form_requested_as = REQUEST_TYPES[form_request_type]
                request = {
                    "id": st.session_state.get("form_id", "NEW-1"),
                    "applicant": {
                        "name": form_name,
                        "organisation": form_org,
                        "country": form_country,
                    },
                    "requested_as": form_requested_as,
                    "requested_format": (
                        "anonymised statistics"
                        if form_requested_as == "health data request"
                        else "pseudonymised records"
                    ),
                    "purpose": form_purpose,
                    "question": form_question,
                    "definitions": [
                        definition.strip()
                        for definition in form_definitions.splitlines()
                        if definition.strip()
                    ],
                }
                with st.spinner("Assessing request..."):
                    resp = process_request(request)
                store(resp, request)
                st.session_state["last"] = resp["request_id"]
        if (
            source != "Fill in a form (challenge format)"
            and request
            and st.button("Submit this request", type="primary")
        ):
            with st.spinner("Assessing request..."):
                resp = process_request(request)
            store(resp, request)
            st.session_state["last"] = resp["request_id"]
    last = st.session_state.get("decisions", {}).get(st.session_state.get("last"))
    if last:
        st.subheader(f"Decision for {last['request_id']}")
        show_response(last)
    if st.session_state.pop("builder_rerun_needed", False):
        st.rerun()


def page_staff():
    st.header("Staff dashboard")
    if st.button("Process all example requests"):
        for req in example_requests().values():
            store(process_request(req), req)
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
