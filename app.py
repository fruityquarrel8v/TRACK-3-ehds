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
from ehds_hdab import disclosure
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
        if resp.get("safe_alternatives"):
            st.markdown("**What you can do instead:**")
            for alternative in resp["safe_alternatives"]:
                st.markdown(f"- {alternative}")
    elif resp["decision"] == "APPROVE_STATISTICAL" and answer:
        st.markdown("**Result (anonymised statistics)**")
        st.dataframe(answer_to_frame(answer, (resp.get("query_plan") or {}).get("group_by", [])), hide_index=True, width="stretch")
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


EXAMPLE_QUESTIONS = [
    ("Explore diabetes prevalence", "How many adults have type 2 diabetes, by region?"),
    ("Compare HbA1c between age groups",
     "Compare HbA1c levels between age groups in patients with type 2 diabetes during 2025."),
    ("Investigate heart failure hospitalisations",
     "How many patients were hospitalised for heart failure in 2025, by hospital?"),
    ("Explore medication use", "How many adults with atrial fibrillation take an oral anticoagulant, by sex?"),
]
DECISION_EXPLANATIONS = {
    "REJECT": "Your request cannot be approved. EHDS Article 54 prohibits this purpose, so no data is released.",
    "APPROVE_STATISTICAL": "Your request is approved for anonymised statistics. Only aggregated, privacy-checked "
                           "numbers are released; groups with fewer than 10 patients are hidden.",
    "APPROVE_SPE": "Your request is approved for a secure processing environment: a locked workspace where approved "
                   "researchers analyse pseudonymised records on site. Nothing can be downloaded; only aggregated "
                   "results leave after review.",
}
STYLE = """
<style>
.block-container {max-width: 1040px; padding-top: 2.2rem;}
.hero {text-align: center; margin: 1.5rem 0 1rem 0;}
.hero h1 {font-size: 2.5rem; font-weight: 700; color: #2B2D33; margin-bottom: .4rem;}
.hero p {color: #6B6F76; font-size: 1.08rem; margin: 0 auto 1rem auto; max-width: 640px;}
div[data-testid="stTextArea"] textarea {border-radius: 16px; font-size: 1.08rem; padding: 1rem;}
div.stButton > button {border-radius: 12px; padding: .5rem 1.1rem;}
.card {background: #FFFFFF; border: 1px solid #ECEBF5; border-radius: 16px; padding: 1.2rem 1.5rem;
       box-shadow: 0 2px 12px rgba(60, 60, 90, .06); margin-bottom: 1rem;}
.card h3 {margin: 0 0 .7rem 0; font-size: 1.15rem; color: #2B2D33;}
.card .row {display: flex; gap: 1rem; padding: .45rem 0; border-bottom: 1px solid #F1F0F7;}
.card .row:last-child {border-bottom: none;}
.card .k {min-width: 150px; color: #6B6F76; font-size: .92rem; padding-top: .1rem;}
.card .v {color: #2B2D33; font-weight: 600;}
.card .d {display: block; color: #8A8E96; font-size: .82rem; font-weight: 400; margin-top: .1rem;}
.steps {color: #8A8E96; margin: .2rem 0 1rem 0; font-size: .95rem;}
.steps b {color: #5B5FC7;}
.quiet {color: #8A8E96; font-size: .85rem;}
</style>
"""


def _go(stage):
    st.session_state["stage"] = stage
    st.session_state["flow_rerun"] = True


def _draft():
    return st.session_state.setdefault("draft", {})


def _escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _card(title, rows):
    body = "".join(
        f'<div class="row"><div class="k">{_escape(k)}</div><div class="v">{_escape(v)}'
        + (f'<span class="d">{_escape(d)}</span>' if d else "") + "</div></div>"
        for k, v, d in rows
    )
    st.markdown(f'<div class="card"><h3>{_escape(title)}</h3>{body}</div>', unsafe_allow_html=True)


def _period_phrase(start, end):
    if start and end and start[:4] == end[:4] and start.endswith("-01-01") and end.endswith("-12-31"):
        return start[:4]
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"since {start}"
    if end:
        return f"up to {end}"
    return ""


def _interpretation_rows(criteria):
    """Plain-language rows for the understanding/review cards: (label, value, detail)."""
    rows = []
    population = criteria.get("population", {}) if isinstance(criteria, dict) else {}
    leaves, nested = _criterion_leaves(population.get("items", []) if isinstance(population, dict) else [])
    if str(population.get("match", "all")).casefold() == "any" and len(leaves) > 1:
        rows.append(("Combination", "Patients meeting at least one of the criteria", "OR"))
    for item in leaves:
        kind = item.get("type")
        excluded = bool(item.get("exclude"))
        prefix = "Excluding " if excluded else ""
        label = str(item.get("label") or kind)
        if kind == "condition":
            when = _period_phrase(item.get("onset_start"), item.get("onset_end"))
            if item.get("onset_start") and item.get("onset_end"):
                when = f"newly diagnosed in {when}"
            elif item.get("onset_start"):
                when = f"diagnosed {when}"
            elif item.get("onset_end"):
                when = f"diagnosed {when}"
            value = f"{prefix}people with {label}" + (f", {when}" if when else "")
            rows.append(("Population", value, f"Identified using ICD-10 {', '.join(item.get('icd10', []))} codes"))
        elif kind == "hospital_stay":
            when = _period_phrase(item.get("period_start"), item.get("period_end"))
            codes = item.get("icd10") or []
            value = f"{prefix}hospital stays" + (f" for {label}" if codes else "") + (f" in {when}" if when else "")
            detail = (f"Inpatient stays with main diagnosis ICD-10 {', '.join(codes)}; emergency visits not counted"
                      if codes else "Inpatient stays of any main diagnosis")
            rows.append(("Hospital stays", value, detail))
        elif kind == "medication":
            status = {"ever": "ever prescribed", "during_period": "prescribed during the period"}.get(
                item.get("status"), "active prescription on 31 Dec 2025")
            value = f"{prefix}taking {label}" if not excluded else f"Not taking {label}"
            rows.append(("Medication", f"{value} ({status})", f"ATC codes {', '.join(item.get('atc', []))}"))
        elif kind == "lab":
            op = {">": "above", ">=": "at least", "<": "below", "<=": "at most"}.get(item.get("op"))
            when = _period_phrase(item.get("period_start"), item.get("period_end"))
            if op:
                value = f"{prefix}latest {label} {op} {item.get('value')} {item.get('unit') or ''}".strip()
            else:
                value = f"{prefix}with a {label} measurement"
            rows.append(("Measurement", value + (f" ({when})" if when else ""),
                         f"LOINC {item.get('loinc')}; latest value per patient"))
        elif kind == "age":
            low, high = item.get("min"), item.get("max")
            value = f"{low}–{high} years" if low is not None and high is not None else (
                f"{low} years or older" if low is not None else f"{high} years or younger")
            rows.append(("Age", prefix + value, "Age on 31 December 2025"))
        elif kind == "sex":
            rows.append(("Sex", prefix + str(item.get("value")).capitalize(), ""))
        elif kind == "region":
            rows.append(("Region", prefix + ", ".join(item.get("values", [])), ""))
        elif kind == "data_holder":
            rows.append(("Hospital", prefix + ", ".join(item.get("values", [])), "Patients registered at this data holder"))
    if nested:
        rows.append(("Combination", "Contains nested criteria groups", "Edit details is not available for nested groups"))
    group_by = criteria.get("group_by", []) if isinstance(criteria, dict) else []
    if group_by:
        names = [DIM_LABELS.get(g.get("dimension"), g.get("dimension")) for g in group_by]
        detail = ""
        for g in group_by:
            if g.get("dimension") == "age_band":
                detail = "Age bands: " + ", ".join(b["label"] for b in g.get("bands", []))
        rows.append(("Compare by", " and ".join(names), detail))
    period = _criteria_period_summary(criteria) if isinstance(criteria, dict) else ""
    if period and period != "No time period applied.":
        rows.append(("Period", period, ""))
    metrics = criteria.get("metrics", []) if isinstance(criteria, dict) else []
    means = [m for m in metrics if m.get("type") == "mean_latest_lab"]
    if means:
        analysis = "Statistical comparison: " + ", ".join(m.get("label", "mean") for m in means)
        if not group_by:
            analysis = "Mean values: " + ", ".join(m.get("label", "mean") for m in means)
    else:
        analysis = "Patient counts" + (" per group" if group_by else "")
    rows.append(("Analysis", analysis, "Counts below 10 are hidden; averages need at least 10 patients"))
    record_types = criteria.get("record_types", []) if isinstance(criteria, dict) else []
    if record_types:
        rows.append(("Records requested", ", ".join(record_types), "Only inside a secure processing environment"))
    return rows


def _pending_clarifications(draft):
    criteria = draft.get("criteria") or {}
    leaves, _ = _criterion_leaves(criteria.get("population", {}).get("items", []))
    answered = draft.setdefault("clarified", {})
    return [
        item for item in leaves
        if item.get("type") == "medication"
        and any(f"'{item.get('label')}'" in a for a in draft.get("assumptions", []))
        and item.get("label") not in answered
    ]


def _screen_welcome(examples):
    st.markdown('<div class="hero"><h1>What would you like to research?</h1>'
                "<p>Describe your research question in your own words. We'll help you turn it into a "
                "structured research request.</p></div>", unsafe_allow_html=True)
    if "pending_question" in st.session_state:
        st.session_state["research_question"] = st.session_state.pop("pending_question")
    question = st.text_area("Your research question", key="research_question", height=140,
                            placeholder="Show HbA1c trends in patients with type 2 diabetes.",
                            label_visibility="collapsed")
    left, _ = st.columns([1, 3])
    if left.button("Continue", type="primary", key="welcome_continue", width="stretch"):
        if not question.strip():
            st.warning("Please describe what you would like to research, or pick an example below.")
        else:
            interpretation = interpret_question(question)
            st.session_state["draft"] = {
                "question": question, "criteria": interpretation.criteria,
                "assumptions": list(interpretation.assumptions),
                "unrecognised": list(interpretation.unrecognised),
                "ok": interpretation.ok(), "source": "question", "clarified": {},
            }
            _go("understand")
    st.caption("Try an example:")
    for col, (label, text) in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(label, key=f"example_{_slug_key(label)}", width="stretch"):
            st.session_state["pending_question"] = text
            _go("welcome")
    st.markdown("")
    st.markdown("**Prefer step-by-step guidance?**")
    if st.button("Build your request with me", key="start_guided"):
        st.session_state["guided_step"] = 1
        _go("guided")
    st.markdown("")
    with st.expander("Advanced options", expanded=False):
        st.caption("For technical users: submit requests in the challenge JSON format.")
        _advanced_options(examples)
    st.markdown(f'<p class="quiet">{PROTOTYPE_DISCLAIMER} Synthetic data only.</p>', unsafe_allow_html=True)


def _slug_key(text):
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _screen_understand():
    draft = _draft()
    criteria = draft.get("criteria") or {}
    st.markdown("### Here's how I understood your research question")
    st.caption(f"\u201c{draft.get('question', '')}\u201d")
    rows = _interpretation_rows(criteria) if draft.get("ok") else []
    if rows:
        _card("Your research request", rows)
    if draft.get("assumptions"):
        st.markdown("**I assumed:**")
        for a in draft["assumptions"]:
            st.markdown(f"- {a}")
    if draft.get("unrecognised"):
        st.warning("I could not match these words to the available data: "
                   + ", ".join(f"\u201c{w}\u201d" for w in draft["unrecognised"])
                   + ". They were not used. The data covers 15 diagnoses, 17 medication groups and 4 laboratory "
                     "tests; use *Edit details* to choose from them.")
    if not draft.get("ok"):
        st.error("I could not find a condition, medication, measurement or patient group in your question that the "
                 "data supports. Rephrase it or build the request step by step.")
    st.caption("Interpretation is rule-based (no AI model) and uses the clinical definitions documented for this "
               "prototype. Patient data never leaves this computer.")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    if draft.get("ok") and c1.button("Looks good — Continue", type="primary", key="understand_continue",
                                     width="stretch"):
        _go("clarify" if _pending_clarifications(draft) else "review")
    if c2.button("Edit details", key="understand_edit", width="stretch"):
        _prefill_builder(criteria, {}, "", list(REQUEST_TYPES)[0])
        st.session_state["guided_step"] = 2
        draft["source"] = "guided"
        _go("guided")
    if c3.button("Back", key="understand_back", width="stretch"):
        _go("welcome")


def _screen_clarify():
    draft = _draft()
    pending = _pending_clarifications(draft)
    st.markdown("### One quick question")
    if not pending:
        _go("review")
        return
    choices = {}
    for item in pending:
        label = item.get("label")
        st.markdown(f"When you say **taking {label}**, which patients do you mean?")
        choices[label] = st.radio(
            label, ["Active prescription on 31 December 2025 (currently treated)",
                    "Ever prescribed (any prescription in the records, 2023–2025)"],
            key=f"clarify_{_slug_key(str(label))}", label_visibility="collapsed")
        st.caption("Active prescriptions follow the challenge's reference date. 'Ever prescribed' also counts "
                   "treatments that have stopped.")
    c1, c2 = st.columns([1, 1])
    if c1.button("Continue", type="primary", key="clarify_continue", width="stretch"):
        leaves, _ = _criterion_leaves(draft["criteria"]["population"]["items"])
        for item in leaves:
            if item.get("type") == "medication" and item.get("label") in choices:
                status = "ever" if choices[item["label"]].startswith("Ever") else "active"
                item["status"] = status
                draft["clarified"][item["label"]] = status
        draft["assumptions"] = [a for a in draft["assumptions"] if not a.startswith("Medication status for")]
        _go("review")
    if c2.button("Back", key="clarify_back", width="stretch"):
        _go("understand")


def _screen_review():
    draft = _draft()
    criteria = draft.get("criteria") or {}
    st.markdown("### Your research request is ready for review")
    if draft.get("question"):
        st.caption(f"\u201c{draft['question']}\u201d")
    _card("Summary", _interpretation_rows(criteria))
    st.info(describe_criteria(criteria))
    if draft.get("assumptions"):
        st.markdown("**Assumptions applied:**")
        for a in draft["assumptions"]:
            st.markdown(f"- {a}")
    st.markdown("#### Who is submitting this request?")
    st.caption("These details are required by the EHDS process and help determine the data-access pathway. "
               "The purpose is checked against the permitted (Art. 53) and prohibited (Art. 54) uses.")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Your name", key="review_name")
    org = c2.text_input("Organisation", key="review_org")
    country = c3.text_input("Country", "AT", key="review_country")
    purpose = st.text_area("Research purpose", key="review_purpose",
                           placeholder="e.g. Public-health research on diabetes care quality to plan prevention programmes.")
    request_type = st.selectbox("What do you need?", list(REQUEST_TYPES), key="review_request_type")
    requested_as = REQUEST_TYPES[request_type]
    record_types = []
    if requested_as == "health data access application":
        record_types = st.multiselect("Records to analyse in the secure processing environment",
                                      ["conditions", "medications", "observations", "encounters"],
                                      key="review_record_types")
        st.caption("Explain in the purpose why anonymised statistics are not enough (e.g. training a prediction "
                   "model on individual histories). Otherwise the request is downgraded to statistics.")
    criteria = dict(criteria)
    criteria["record_types"] = record_types
    b1, b2, b3 = st.columns([1.4, 1, 1])
    if b1.button("Submit research request", type="primary", key="review_submit", width="stretch"):
        missing = [label for label, value in (("your name", name), ("organisation", org), ("research purpose", purpose))
                   if not str(value).strip()]
        if missing:
            st.error("Please add " + ", ".join(missing) + " before submitting.")
        else:
            number = st.session_state.get("flow_request_number", 0) + 1
            st.session_state["flow_request_number"] = number
            prefix = "GUIDED" if draft.get("source") == "guided" else "NL"
            request = {
                "id": f"{prefix}-{number}",
                "applicant": {"name": name, "organisation": org, "country": country},
                "requested_as": requested_as,
                "requested_format": "anonymised statistics" if requested_as == "health data request"
                else "pseudonymised records",
                "purpose": purpose,
                "question": draft.get("question") or describe_criteria(criteria),
                "definitions": criteria_to_definitions(criteria),
                "criteria": criteria,
            }
            with st.spinner("Assessing your request..."):
                resp = process_request(request)
            store(resp, request)
            st.session_state["last"] = resp["request_id"]
            _go("result")
    if b2.button("Edit details", key="review_edit", width="stretch"):
        _prefill_builder(criteria, {"name": name, "organisation": org, "country": country}, purpose, request_type)
        st.session_state["guided_step"] = 2
        draft["source"] = "guided"
        _go("guided")
    if b3.button("Back", key="review_back", width="stretch"):
        _go("guided" if draft.get("source") == "guided" else "understand")
    with st.expander("Technical details (request JSON)", expanded=False):
        st.json({"question": draft.get("question"), "criteria": criteria})


def _screen_result():
    resp = st.session_state.get("decisions", {}).get(st.session_state.get("last"))
    if not resp:
        _go("welcome")
        return
    approved = resp["decision"] != "REJECT"
    st.markdown("### Your research results" if approved else "### Decision on your request")
    st.write(DECISION_EXPLANATIONS.get(resp["decision"], ""))
    if resp.get("decision") == "APPROVE_STATISTICAL" and resp.get("answer") and not resp.get("withheld"):
        plan = resp.get("query_plan") or {}
        if not plan.get("group_by"):
            value = next(iter(next(iter(resp["answer"].values()), {}).values()), None)
            if value is not None:
                st.metric("Eligible patients", str(value))
    show_response(resp)
    request = st.session_state.get("requests", {}).get(resp["request_id"]) or {}
    if isinstance(request.get("criteria"), dict):
        with st.expander("How this was defined", expanded=False):
            _card("Definitions used", _interpretation_rows(request["criteria"]))
    if st.button("Ask another question", key="result_new"):
        st.session_state["draft"] = {}
        st.session_state["pending_question"] = ""
        _go("welcome")


GUIDED_STEPS = ["Research question", "Population", "Analysis", "Review"]
GUIDED_GOALS = ["Explore a medical condition", "Compare patient groups", "Study laboratory measurements",
                "Investigate medication use", "Explore hospitalisations", "Something else"]


def _progress(step):
    parts = [f"<b>{i}. {name}</b>" if i == step else f"{i}. {name}" for i, name in enumerate(GUIDED_STEPS, 1)]
    st.markdown('<div class="steps">' + " &rarr; ".join(parts) + "</div>", unsafe_allow_html=True)


def _population_step(data):
    """Criteria rows, time period and ALL/ANY. Returns (items, match, period_start, period_end, invalid)."""
    years = set()
    for table, column in (("conditions", "onset_date"), ("observations", "date"),
                          ("medications", "start_date"), ("encounters", "start_date")):
        years.update(int(v) for v in data[table][column].dropna().dt.year.unique() if int(v) <= 2025)
    year_options = [None, *sorted(years)]
    st.markdown("**Time period** (optional)")
    y1, y2 = st.columns(2)
    from_year = y1.selectbox("From year", year_options, key="builder_from_year",
                             format_func=lambda v: "No start year" if v is None else str(v))
    to_year = y2.selectbox("To year", year_options, key="builder_to_year",
                           format_func=lambda v: "No end year" if v is None else str(v))
    st.caption("Lab, medication and hospital-stay records cover 2023–2025; diagnoses go back further.")
    period_start, period_end = _year_bounds(from_year, to_year)
    invalid_period = from_year is not None and to_year is not None and from_year > to_year
    if invalid_period:
        st.error("The start year must not be after the end year.")
    st.markdown("**Patients must meet:**")
    st.radio("Patients must meet", ["ALL of the criteria", "ANY of the criteria"], horizontal=True,
             key="builder_match", label_visibility="collapsed",
             help="ALL = every criterion applies (AND). ANY = at least one applies (OR). "
                  "Use 'Exclude' on a criterion for NOT.")
    categories = ["Diagnosis", "Medication", "Laboratory measurement", "Age", "Sex", "Region",
                  "Hospital (data holder)", "Hospital stay (main diagnosis)"]
    a1, a2 = st.columns([2, 1])
    category = a1.selectbox("Add a criterion", categories, key="builder_new_category")
    if a2.button("Add criterion", key="builder_add_criterion", width="stretch"):
        row_id = st.session_state.get("builder_next_criterion_id", 0) + 1
        st.session_state["builder_next_criterion_id"] = row_id
        st.session_state.setdefault("builder_criteria", []).append({"id": row_id, "category": category})
    st.session_state.setdefault("builder_criteria", [])
    items = []
    for row in list(st.session_state["builder_criteria"]):
        with st.container(border=True):
            criterion = _builder_criterion_row(row, data, period_start, period_end)
        if criterion is not None:
            items.append(criterion)
    if not st.session_state["builder_criteria"]:
        st.caption("No criteria yet. Add at least one, e.g. a diagnosis.")
    match = "any" if st.session_state.get("builder_match") == "ANY of the criteria" else "all"
    return items, match, period_start, period_end, invalid_period


def _analysis_step(data, period_start, period_end):
    """Returns (group_by, metrics)."""
    st.markdown("**What would you like to find out?**")
    analysis = st.radio("Analysis", ["Count patients", "Compare groups", "Calculate a mean", "Explore changes over time"],
                        key="guided_analysis", label_visibility="collapsed")
    explanations = {
        "Count patients": "How many patients meet your criteria.",
        "Compare groups": "See how results differ between men and women, age bands, regions or hospitals.",
        "Calculate a mean": "The average of each patient's latest laboratory value (within the period if set).",
        "Explore changes over time": "Not available yet in this prototype: results cannot be grouped by year or month. "
                                     "You can compare groups instead.",
    }
    st.caption(explanations[analysis])
    group_by = []
    if analysis in ("Compare groups", "Calculate a mean"):
        choices = {"Sex": "sex", "Age band": "age_band", "Region": "region", "Hospital": "data_holder"}
        selected = st.multiselect("Compare by", list(choices), key="builder_group_by")
        for label in selected:
            dimension = choices[label]
            if dimension == "age_band":
                preset = st.selectbox("Age bands", ["10-year bands", "18–39 / 40–64 / 65+", "Custom"],
                                      key="builder_age_band_preset")
                custom = st.text_input("Custom age bands", "40-49, 50-59, 60+",
                                       key="builder_custom_age_bands") if preset == "Custom" else ""
                bands = _age_bands(preset, custom)
                if bands:
                    group_by.append({"dimension": "age_band", "bands": bands})
                else:
                    st.warning("Enter valid age bands such as 40-49, 50-59, 60+.")
            else:
                group_by.append({"dimension": dimension})
    metrics = [{"type": "count_patients", "label": "patients"}]
    want_mean = analysis == "Calculate a mean" or (
        analysis == "Compare groups" and st.checkbox("Also compare a laboratory value", key="builder_mean_enabled"))
    if want_mean:
        lab_options = _builder_concepts(data, "observation")
        selected_lab = st.selectbox("Laboratory value", [o["label"] for o in lab_options], key="builder_mean_lab")
        lab = next(o for o in lab_options if o["label"] == selected_lab)
        metric = {"type": "mean_latest_lab", "label": f"mean latest {lab['term']} ({lab['unit']})",
                  "loinc": lab["codes"][0]}
        if period_start:
            metric["period_start"] = period_start
        if period_end:
            metric["period_end"] = period_end
        metrics.append(metric)
    return group_by, metrics


def _screen_guided():
    data = warm_data()
    draft = _draft()
    step = st.session_state.get("guided_step", 1)
    st.markdown("### Build your request step by step")
    _progress(step)
    if step == 1:
        st.markdown("**What would you like to investigate?**")
        st.radio("Goal", GUIDED_GOALS, key="guided_goal", label_visibility="collapsed")
        st.text_area("Tell me more (optional)", key="guided_details",
                     placeholder="e.g. whether blood-sugar control differs between hospitals")
        c1, c2 = st.columns([1, 1])
        if c1.button("Next", type="primary", key="guided_next_1", width="stretch"):
            st.session_state["guided_step"] = 2
            _go("guided")
        if c2.button("Back to start", key="guided_back_1", width="stretch"):
            _go("welcome")
    elif step == 2:
        st.markdown("**Which patients would you like to include?**")
        items, match, period_start, period_end, invalid = _population_step(data)
        st.session_state["guided_items"] = items
        if items:
            st.info(describe_criteria({"population": {"match": match, "items": items}}))
        c1, c2 = st.columns([1, 1])
        if c1.button("Next", type="primary", key="guided_next_2", width="stretch",
                     disabled=not items or invalid):
            st.session_state["guided_step"] = 3
            _go("guided")
        if c2.button("Back", key="guided_back_2", width="stretch"):
            st.session_state["guided_step"] = 1
            _go("guided")
    elif step == 3:
        period_start, period_end = _year_bounds(st.session_state.get("builder_from_year"),
                                                st.session_state.get("builder_to_year"))
        group_by, metrics = _analysis_step(data, period_start, period_end)
        c1, c2 = st.columns([1, 1])
        if c1.button("Next", type="primary", key="guided_next_3", width="stretch"):
            # Recompute the population from the stored rows without rendering them again.
            items = st.session_state.get("guided_items", [])
            match = "any" if st.session_state.get("builder_match") == "ANY of the criteria" else "all"
            criteria = {"population": {"match": match, "items": items}, "group_by": group_by,
                        "metrics": metrics, "record_types": []}
            goal = st.session_state.get("guided_goal", "")
            details = st.session_state.get("guided_details", "")
            draft.update({"criteria": criteria, "assumptions": [], "unrecognised": [], "ok": bool(items),
                          "source": "guided", "clarified": {},
                          "question": (details.strip() or goal) if (details or goal) else describe_criteria(criteria)})
            st.session_state["guided_step"] = 4
            _go("review")
        if c2.button("Back", key="guided_back_3", width="stretch"):
            st.session_state["guided_step"] = 2
            _go("guided")


PERSISTED_PREFIXES = ("builder_", "guided_", "review_", "clarify_", "research_question")
_BUTTON_KEY = re.compile(r"(?:_submit|_edit|_back|_continue|_remove|_add_criterion|_next_\d+|_back_\d+)$")


def _keep_widget_state():
    """Streamlit forgets a keyed widget's value when the widget is not rendered (e.g. on another wizard
    step). Re-assigning the keys each run keeps Back/Next from losing answers."""
    for key in list(st.session_state.keys()):
        if (isinstance(key, str) and key.startswith(PERSISTED_PREFIXES)
                and not _BUTTON_KEY.search(key)):
            st.session_state[key] = st.session_state[key]


def page_applicant():
    st.markdown(STYLE, unsafe_allow_html=True)
    _keep_widget_state()
    examples = example_requests()
    stage = st.session_state.get("stage", "welcome")
    if stage == "welcome":
        _screen_welcome(examples)
    elif stage == "understand":
        _screen_understand()
    elif stage == "clarify":
        _screen_clarify()
    elif stage == "review":
        _screen_review()
    elif stage == "result":
        _screen_result()
    elif stage == "guided":
        _screen_guided()
    if st.session_state.pop("builder_rerun_needed", False) or st.session_state.pop("flow_rerun", False):
        st.rerun()


def _advanced_options(examples):
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
            _go("result")
    if (
        source != "Fill in a form (challenge format)"
        and request
        and st.button("Submit this request", type="primary")
    ):
        with st.spinner("Assessing request..."):
            resp = process_request(request)
        store(resp, request)
        st.session_state["last"] = resp["request_id"]
        _go("result")


DEMO_APPLICANT = {"name": "Dr. Demo Researcher", "organisation": "Vienna Cancer Research Group (university)",
                  "country": "AT"}
DEMO_SCENARIOS = [
    ("1. A legitimate question", "How many women with breast cancer are there in Austria?",
     "Scientific research on breast cancer care to plan regional screening services.",
     "The assistant interprets the question, checks the purpose and releases an anonymised count."),
    ("2. The same question, one region removed",
     "How many women with breast cancer are there in Austria, excluding Burgenland?",
     "Scientific research on breast cancer care to plan regional screening services.",
     "Both counts pass the small-cell rule on their own, but their difference would reveal how many "
     "patients live in the excluded region. The ledger notices and withholds the result."),
    ("3. An ambiguous question",
     "How many adults with atrial fibrillation take an oral anticoagulant?",
     "Public health research on anticoagulation in atrial fibrillation.",
     "\"Take\" does not say whether the prescription must be current. The assistant asks instead of guessing."),
    ("4. A prohibited purpose", "How many adults have type 2 diabetes, by region?",
     "We want to adjust health insurance premiums for people with diabetes by region.",
     "The purpose falls under EHDS Art. 54, so the request is rejected before any data is touched."),
]


def page_demo():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.header("Demo scenarios")
    st.write("Four prepared scenarios for the judges. Every button runs the real pipeline on the synthetic data; "
             "nothing is pre-recorded. Message: *following a privacy rule is not always the same as protecting privacy.*")
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Disclosure ledger: {len(disclosure.ledger_summary())} recorded cells. Scenario 2 only triggers "
               "after scenario 1 has released its count.")
    if c2.button("Reset demo state", key="demo_reset"):
        disclosure.reset_ledger()
        for key in list(st.session_state.get("decisions", {})):
            if str(key).startswith("DEMO-"):
                st.session_state["decisions"].pop(key, None)
        st.success("Ledger reset.")
    for index, (title, question, purpose, message) in enumerate(DEMO_SCENARIOS, start=1):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(f"Question: \u201c{question}\u201d  \nPurpose: \u201c{purpose}\u201d")
            st.caption(message)
            interpretation = interpret_question(question)
            if index == 3:
                st.info("Clarification the assistant asks: \"When you say *taking Oral anticoagulant*, do you mean "
                        "an active prescription on 31 December 2025 or anyone ever prescribed it?\" "
                        "Assumption if unanswered: " + "; ".join(interpretation.assumptions))
            if st.button("Run scenario", key=f"demo_run_{index}"):
                request = {
                    "id": f"DEMO-{index}", "applicant": DEMO_APPLICANT,
                    "requested_as": "health data request", "requested_format": "anonymised statistics",
                    "purpose": purpose, "question": question,
                    "definitions": criteria_to_definitions(interpretation.criteria),
                    "criteria": interpretation.criteria,
                }
                with st.spinner("Running through the real pipeline..."):
                    resp = process_request(request)
                store(resp, request)
            resp = st.session_state.get("decisions", {}).get(f"DEMO-{index}")
            if resp:
                st.write(DECISION_EXPLANATIONS.get(resp["decision"], ""))
                show_response(resp)


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
        st.dataframe(overview, hide_index=True, width="stretch")
        rid = st.selectbox("Inspect request", list(decisions))
        show_response(decisions[rid], staff=True)
    st.subheader("Disclosure ledger")
    st.caption("Every released count is remembered (with one-way hashes of the patient IDs, never the IDs) so that "
               "a new result that would reveal a small group by subtraction can be withheld.")
    ledger = disclosure.ledger_summary()
    if ledger:
        rows = [{"request": e.get("request_id"), "cell": " / ".join(map(str, e.get("path") or [])),
                 "metric": e.get("metric"), "released value": e.get("released"), "status": e.get("status"),
                 "time": str(e.get("timestamp", ""))[:19]} for e in ledger[::-1]]
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    else:
        st.caption("Empty.")
    with st.expander("Administrative actions (demo only)"):
        st.caption("Resetting the ledger forgets what has been released. Only for demonstrations; a production "
                   "HDAB would never reset its disclosure history.")
        if st.button("Reset disclosure ledger", key="staff_reset_ledger"):
            disclosure.reset_ledger()
            st.success("Disclosure ledger reset.")
    st.subheader("Audit log")
    log = Path(config.AUDIT_LOG)
    if log.exists():
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(lines[::-1]).astype(str), hide_index=True, width="stretch")
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
PAGES = {"Research assistant": page_applicant, "Demo scenarios": page_demo, "Staff dashboard": page_staff,
         "Secure processing environment": page_spe, "Evaluation": page_evaluation}
st.sidebar.title("EHDS Health Data Access Body")
st.sidebar.caption("Simulated HDAB for AIM Hackathon Vienna 2026, Track 3. Synthetic data only.")
st.sidebar.caption(PROTOTYPE_DISCLAIMER)
PAGES[st.sidebar.radio("View", list(PAGES))]()
