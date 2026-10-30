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
from ehds_hdab.providers import interpret as interpret_question
from ehds_hdab.pipeline import process_request
from ehds_hdab.charts import build_plan, altair_chart, display_table, png_table_and_spec, select_chart_type
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

st.set_page_config(page_title="AIM Health Data Navigator", page_icon=str(ROOT / "assets" / "aim_mark.png"), layout="wide")


@st.cache_resource
def warm_data():
    return load_data()


def example_requests():
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(REQUESTS_DIR.glob("R*.json"))}


DIM_LABELS = {"year": "Year", "region": "Region", "age_band": "Age band", "data_holder": "Hospital", "sex": "Sex"}


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
    if resp.get("withheld"):
        kind, title = "warning", "Approved request, result withheld by the disclosure check"
    getattr(st, kind)(f"**{title}** (`{resp['decision']}`)")
    st.markdown(f"**Reason:** {resp['reason']}")
    st.markdown("**Legal basis:**")
    show_articles(resp.get("articles"))
    if resp.get("needs_staff_review") and not resp.get("withheld"):
        st.warning("Parts of this request could not be interpreted automatically. A staff member will review it.")
    elif resp.get("withheld"):
        st.caption("A staff member will review this release decision.")
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


def _derived_request(resp, result):
    """Copy of the original request with a new grouping and/or an extra filter; goes through the full pipeline."""
    request = st.session_state.get("requests", {}).get(resp["request_id"])
    if not request or not isinstance(request.get("criteria"), dict):
        st.info("Regrouping and filtering are available for requests made with the research question box or the "
                "guided builder.")
        return None
    derived = copy.deepcopy(request)
    if result.get("regroup"):
        existing_years = next((d for d in request["criteria"].get("group_by", []) or []
                               if isinstance(d, dict) and d.get("dimension") == "year"), None)
        derived["criteria"]["group_by"] = [
            {"dimension": "age_band", "bands": _age_bands("10-year bands", "")} if dimension == "age_band"
            else (existing_years or {"dimension": "year"}) if dimension == "year"
            else {"dimension": dimension}
            for dimension in result["regroup"]
        ]
    if result.get("filter"):
        population = derived["criteria"].setdefault("population", {"match": "all", "items": []})
        if str(population.get("match", "all")).casefold() == "any" and len(population.get("items", [])) > 1:
            population["items"] = [{"type": "group", "match": "any", "items": population["items"]}]
            population["match"] = "all"
        population["items"] = [item for item in population.get("items", []) if item.get("type") != "age"]
        population["items"].append(dict(result["filter"]))
    number = st.session_state.get("regroup_number", 0) + 1
    st.session_state["regroup_number"] = number
    derived["id"] = f"{str(resp['request_id']).split('-G')[0]}-G{number}"
    derived["parent_id"] = str(resp["request_id"]).split("-G")[0]
    return derived


def _regroup(resp, result):
    derived = _derived_request(resp, result)
    if derived is None:
        return
    with st.spinner("Re-running the request through the approval process..."):
        new_resp = process_request(derived)
    store(new_resp, derived)
    st.session_state["last"] = new_resp["request_id"]
    st.session_state.setdefault("chart_specs", {})[new_resp["request_id"]] = dict(result.get("spec") or {})
    st.rerun()


CHART_TYPE_LABELS = {"line": "Line graph", "bar": "Bar chart"}
COMPARE_OPTIONS = {"Year": "year", "Sex": "sex", "Age band": "age_band", "Region": "region", "Hospital": "data_holder"}


def _chart_section(resp):
    rid = resp["request_id"]
    request = st.session_state.get("requests", {}).get(rid) or {}
    criteria = request.get("criteria") if isinstance(request.get("criteria"), dict) else None
    population_text = describe_criteria(criteria) if criteria else str(request.get("question") or rid)
    period_text = _criteria_period_summary(criteria) if criteria else "As defined in the request"
    specs = st.session_state.setdefault("chart_specs", {})
    overrides = dict(specs.get(rid) or {})
    group_by = (resp.get("query_plan") or {}).get("group_by", [])
    dims = [DIM_LABELS.get(d.get("dimension"), d.get("dimension")) for d in group_by]
    plan = build_plan(resp, population_text, period_text, overrides)
    if plan is None:
        return
    st.markdown("**Visualise these results**")
    st.caption("Charts are drawn only from the anonymised, privacy-checked table above. Changing the grouping or a "
               "filter re-runs the request through the same approval and disclosure checks.")
    single_number = not dims and not plan["category_axis"]
    if single_number:
        st.caption("A single number is not charted. Ask for a trend by year or a comparison by sex, age, region or "
                   "hospital to get a chart.")
    else:
        c1, c2, c3, c4 = st.columns([1.1, 1.8, 1.4, 0.9])
        type_key = f"chart_type_{rid}"
        if overrides.pop("pending_sync", False):
            # A typed command changed the chart on the previous run; widgets may only be updated before rendering.
            specs[rid] = overrides
            if overrides.get("type") in plan["allowed_types"]:
                st.session_state[type_key] = overrides["type"]
            if overrides.get("metric") in plan["metric_options"]:
                st.session_state[f"chart_metric_{rid}"] = overrides["metric"]
            if overrides.get("table") is not None:
                st.session_state[f"chart_table_{rid}"] = bool(overrides.pop("table"))
        if st.session_state.get(type_key) not in plan["allowed_types"]:
            st.session_state[type_key] = plan["chart_type"]
        chart_type = c1.selectbox("Chart type", plan["allowed_types"], key=type_key,
                                  format_func=lambda t: CHART_TYPE_LABELS.get(t, t))
        metric_key = f"chart_metric_{rid}"
        if st.session_state.get(metric_key) not in plan["metric_options"]:
            st.session_state[metric_key] = plan["metric"]
        metric = c2.selectbox("Measure", plan["metric_options"], key=metric_key) if len(plan["metric_options"]) > 1 \
            else plan["metric"]
        compare_choices = ["No change"] + [label for label in COMPARE_OPTIONS if label not in dims]
        compare = c3.selectbox("Add a comparison", compare_choices, key=f"chart_compare_{rid}")
        show_table = c4.toggle("Table", key=f"chart_table_{rid}")
        if compare != "No change":
            st.session_state[f"chart_compare_{rid}"] = "No change"
            _regroup(resp, {"regroup": [COMPARE_OPTIONS[d] for d in dims if d in COMPARE_OPTIONS] + [COMPARE_OPTIONS[compare]],
                            "spec": {"type": chart_type, "metric": metric}})
        if chart_type != plan["chart_type"] or metric != plan["metric"]:
            overrides.update({"type": chart_type, "metric": metric})
            specs[rid] = overrides
            plan = build_plan(resp, population_text, period_text, overrides)
    examples = (["Show this by year as a line graph", "Compare by sex", "Compare by region"] if single_number
                else ([f"Line graph of {plan['metric']} by year", "Change this into a bar chart", "Show a table instead"]
                      if plan["time_axis"] else
                      [f"Bar chart of {plan['metric']} by {plan['x_label']}", "Show this by year as a line graph",
                       "Only patients aged 40-65"]))
    command = st.text_input("Refine the chart in your own words", key=f"chart_cmd_{rid}", placeholder=examples[0])
    chosen = None
    for i, (col, example) in enumerate(zip(st.columns(3), examples)):
        if col.button(example, key=f"chart_ex_{rid}_{i}"):
            chosen = example
    command = chosen or command
    message = ""
    if command and command == overrides.get("handled") and not chosen:
        command = ""  # already applied on a previous run; the text box still shows it
    if command:
        overrides["handled"] = command
        specs[rid] = overrides
        result = interpret_chart_command(command, {"type": plan["chart_type"], "metric": plan["metric"],
                                                   "x": plan["x"], "color": plan["color"]},
                                         plan["metric_options"], dims)
        if result.get("regroup") or result.get("filter"):
            _regroup(resp, result)
            return
        if result.get("table"):
            overrides.update({"table": True, "pending_sync": True})
            specs[rid] = overrides
            st.rerun()
        elif result.get("refused_type"):
            _, _, message = select_chart_type(plan["time_axis"], plan["category_axis"], result["refused_type"])
        elif result["ok"]:
            wanted = result["spec"]
            new_type, _, note = select_chart_type(plan["time_axis"], plan["category_axis"], wanted.get("type"))
            overrides.update({"type": new_type, "metric": wanted.get("metric") or plan["metric"],
                              "pending_sync": True, "note": note})
            specs[rid] = overrides
            st.rerun()
        else:
            message = result["message"]
    if overrides.get("note") and not message:
        message = overrides.pop("note")
        specs[rid] = overrides
    if plan.get("selection_note") and not message:
        message = plan["selection_note"]
    if message:
        st.info(message)
    if single_number:
        return
    if show_table if not single_number else False:
        st.dataframe(display_table(plan), hide_index=True, width="stretch")
    else:
        st.altair_chart(altair_chart(plan), width="stretch")
    st.caption(plan["explanation"])
    with st.expander("Data behind this chart"):
        st.dataframe(display_table(plan), hide_index=True, width="stretch")
        st.caption(plan["privacy"])
    with st.expander("Statistical methodology and definitions"):
        st.write(f"**Measure:** {plan['measure_text']}" + (f" ({plan['unit']})" if plan["unit"] else ""))
        st.write(f"**Population:** {population_text}")
        st.write(f"**Time period:** {period_text}")
        st.write(methodology_text({"type": plan["chart_type"], "metric": plan["metric"]}, resp))
    frame, spec = png_table_and_spec(plan)
    try:
        _, png = render_chart(frame, spec, plan["title"], population_text, period_text, "")
        st.download_button("Download chart (PNG)", png, file_name=f"{rid}_chart.png", mime="image/png",
                           key=f"chart_png_{rid}")
    except ValueError:
        pass
    b1, b2 = st.columns([1, 3])
    if b1.button("Restore original chart", key=f"chart_restore_{rid}"):
        specs.pop(rid, None)
        for key in (f"chart_type_{rid}", f"chart_metric_{rid}", f"chart_table_{rid}", f"chart_cmd_{rid}"):
            st.session_state.pop(key, None)
        parent = request.get("parent_id")
        if parent and parent in st.session_state.get("decisions", {}):
            st.session_state["last"] = parent
        st.rerun()


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


def _year_value(value):
    return value if isinstance(value, int) and value > 0 else None


CONNECTORS_FIRST = ["Include", "NOT"]
CONNECTORS = ["AND", "OR", "AND NOT", "OR NOT"]


def _combine_rows(pairs):
    """Turn [(connector, criterion)] read top-to-bottom into population items: OR joins a row with the
    row above (an ANY group); AND starts a new group; NOT excludes that row."""
    groups = []
    for index, (op, item) in enumerate(pairs):
        item = dict(item)
        item["exclude"] = "NOT" in str(op)
        if index == 0 or str(op).startswith("AND") or str(op) in CONNECTORS_FIRST:
            groups.append([item])
        else:
            groups[-1].append(item)
    return [group[0] if len(group) == 1 else {"type": "group", "match": "any", "items": group} for group in groups]


def _rows_from_population(population):
    """Inverse of _combine_rows for one level of nesting: [(connector, leaf)], or None if the structure
    cannot be shown as a top-to-bottom list (deeper nesting, excluded groups, (A AND B) OR C)."""
    if not isinstance(population, dict):
        return []
    items = population.get("items", []) or []
    top_any = str(population.get("match", "all")).casefold() == "any"
    pairs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "items" in item:
            children = [c for c in item.get("items", []) if isinstance(c, dict)]
            if item.get("exclude") or any("items" in c for c in children):
                return None
            group_any = str(item.get("match", "all")).casefold() == "any"
            if top_any and not group_any and len(children) > 1:
                return None
            first, rest = ("OR", "OR") if top_any else (("AND", "OR") if group_any else ("AND", "AND"))
            pairs.extend((first if index == 0 else rest, child) for index, child in enumerate(children))
        else:
            pairs.append(("OR" if top_any else "AND", item))
    return pairs


def _leaf_label(item):
    kind = item.get("type")
    if kind == "age":
        low, high = item.get("min"), item.get("max")
        return f"aged {low}\u2013{high}" if low is not None and high is not None else (
            f"aged {low}+" if low is not None else f"aged \u2264{high}")
    if kind == "sex":
        return str(item.get("value", "")).capitalize()
    if kind in {"region", "data_holder"}:
        return ", ".join(item.get("values", []))
    if kind == "lab" and item.get("op"):
        return f"{item.get('label')} {item.get('op')} {item.get('value')}"
    return str(item.get("label") or kind)


def _boolean_summary(pairs):
    groups = []
    for index, (op, item) in enumerate(pairs):
        text = ("NOT " if ("NOT" in str(op) or item.get("exclude")) else "") + _leaf_label(item)
        if index == 0 or str(op).startswith("AND") or str(op) in CONNECTORS_FIRST:
            groups.append([text])
        else:
            groups[-1].append(text)
    return " AND ".join(f"({' OR '.join(g)})" if len(g) > 1 else g[0] for g in groups)


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
    leaves, _ = _criterion_leaves(items)
    pairs = _rows_from_population(population)
    has_nested_group = pairs is None
    if has_nested_group:
        pairs = [("AND", leaf) for leaf in leaves]
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
    st.session_state["builder_from_year"] = min(starts) if starts else 0
    st.session_state["builder_to_year"] = max(ends) if ends else 0

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
    year_group = next((item for item in group_by if item.get("dimension") == "year"), None)
    if year_group:
        st.session_state["guided_analysis"] = "Explore changes over time"
        st.session_state["builder_year_from"] = int(year_group.get("from") or 2023)
        st.session_state["builder_year_to"] = int(year_group.get("to") or 2025)
    elif st.session_state.get("guided_analysis") == "Explore changes over time":
        st.session_state["guided_analysis"] = "Compare groups" if group_by else "Count patients"
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

    for index, (op, item) in enumerate(pairs, start=1):
        kind = item.get("type")
        if kind not in categories:
            continue
        category, option_kind, code_field, key_suffix = categories[kind]
        row_id = index
        excluded = bool(item.get("exclude"))
        if not st.session_state["builder_criteria"]:
            connector = "NOT" if excluded else "Include"
        else:
            connector = str(op).split()[0] + (" NOT" if excluded else "")
        st.session_state["builder_criteria"].append(
            {"id": row_id, "category": category}
        )
        st.session_state["builder_next_criterion_id"] = row_id
        st.session_state[f"builder_{row_id}_op"] = connector
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


def _builder_criterion_row(row, data, period_start, period_end, first=False):
    row_id = row["id"]
    category = row["category"]
    criterion = None
    options = CONNECTORS_FIRST if first else CONNECTORS
    op_key = f"builder_{row_id}_op"
    current = st.session_state.get(op_key)
    if current not in options:
        st.session_state[op_key] = (options[1] if current and "NOT" in str(current) else options[0])
    op_col, title_col, remove_col = st.columns([1.3, 3.7, 1])
    op = op_col.selectbox("Combine with the rows above", options, key=op_key, label_visibility="collapsed",
                          help="Read top to bottom: AND starts a new group, OR joins this row with the row "
                               "above, NOT excludes patients matching this row.")
    title_col.markdown(f"**{category}**")
    exclude = "NOT" in str(op)
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
    year_group = next((g for g in criteria.get("group_by", []) or []
                       if isinstance(g, dict) and g.get("dimension") == "year"), None)
    if year_group:
        span = (f"{year_group['from']}\u2013{year_group['to']}" if year_group.get("from") and year_group.get("to")
                else "2023\u20132025 or the period given")
        return f"Per calendar year, {span}"
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
:root {--navy: #0E2554; --ink: #22305A; --muted: #5C6690; --blue: #2F56D8; --blue-dark: #2446B3;
       --tint: #F1EFFC; --tint-2: #F7F6FE; --line: #DDD9F3; --green: #1E8E5A; --green-tint: #E8F7EF;
       --lavender: #9A8CF2; --coral: #FF7A6B;}
.block-container {max-width: 1120px; padding-top: 3.4rem;}
h1, h2, h3, h4 {color: var(--navy) !important; letter-spacing: -.01em;}
.hdn-header {display: flex; align-items: center; justify-content: space-between; padding: .2rem 0 1rem 0;
             border-bottom: 1px solid var(--line); margin-bottom: 1.4rem;}
.hdn-brand {display: flex; align-items: center; gap: .9rem;}
.hdn-brand .t {font-size: 1.55rem; font-weight: 800; color: var(--navy); line-height: 1.1;}
.hdn-brand img {height: 56px; width: auto;}
.hdn-brand .s {font-size: .9rem; color: var(--muted); margin-top: .15rem;}
.hdn-eu {display: flex; align-items: center; gap: .6rem; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
         color: #1E3A8A; text-transform: uppercase; line-height: 1.25;}
.eyebrow {font-size: .74rem; font-weight: 700; letter-spacing: .16em; color: #5B6680; text-transform: uppercase;
          margin: .2rem 0 .3rem 0;}
.hero h1 {font-size: 2.35rem; font-weight: 800; color: var(--navy); margin: 0 0 .5rem 0; line-height: 1.15;}
.hero p {color: var(--muted); font-size: 1.05rem; margin: 0 0 1rem 0; max-width: 640px;}
.screen-title {font-size: 1.7rem; font-weight: 800; color: var(--navy); margin: 0 0 .3rem 0;}
.screen-sub {color: var(--muted); margin: 0 0 1rem 0;}
div[data-testid="stTextArea"] textarea {border-radius: 12px; border: 1px solid var(--line); font-size: 1.05rem;
                                         padding: 1rem; background: #fff;}
div.stButton > button, div.stDownloadButton > button {border-radius: 10px; padding: .55rem 1.15rem; font-weight: 600;
                                                       border: 1px solid var(--line); color: var(--navy);}
div.stButton > button[kind="primary"] {background: var(--blue); border-color: var(--blue); color: #fff;}
div.stButton > button[kind="primary"]:hover {background: var(--blue-dark); border-color: var(--blue-dark);}
.card {background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.3rem;
       box-shadow: 0 4px 18px rgba(27, 42, 78, .06); margin-bottom: 1rem;}
.card .head {display: flex; align-items: center; gap: .7rem; margin-bottom: .25rem;}
.card .head .chk {width: 30px; height: 30px; border-radius: 50%; background: var(--green-tint); color: var(--green);
                  display: inline-flex; align-items: center; justify-content: center; flex: none;}
.card h3 {margin: 0; font-size: 1.25rem; color: var(--navy); font-weight: 800;}
.card .lead {color: var(--muted); font-size: .92rem; margin: 0 0 .9rem 0;}
.card .row {display: flex; align-items: center; gap: .9rem; background: var(--tint); border-radius: 12px;
            padding: .7rem .9rem; margin-bottom: .5rem; border: 1px solid transparent;}
.card .row .ic {width: 34px; height: 34px; border-radius: 9px; background: #fff; color: var(--blue); flex: none;
                display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 1px 4px rgba(27,42,78,.08);}
.card .row .k {color: var(--muted); font-size: .8rem; margin-bottom: .1rem;}
.card .row .v {color: var(--navy); font-weight: 700; font-size: 1rem;}
.card .row .d {display: block; color: #6F7A94; font-size: .8rem; font-weight: 400; margin-top: .1rem;}
.banner-ok {display: flex; align-items: center; gap: .8rem; background: var(--green-tint); border-radius: 12px;
            padding: .8rem 1rem; margin: .2rem 0 1rem 0; border: 1px solid #CDEBD9;}
.banner-ok .ic {width: 34px; height: 34px; border-radius: 9px; background: var(--green); color: #fff; flex: none;
                display: inline-flex; align-items: center; justify-content: center;}
.banner-ok b {color: #176B45; display: block; font-size: .95rem;}
.banner-ok span {color: #3F6B55; font-size: .85rem;}
.strip {display: flex; gap: 1.2rem; margin: 1.2rem 0 .6rem 0; flex-wrap: wrap;}
.strip .st {flex: 1 1 200px; min-width: 180px;}
.strip .num {display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px;
             border-radius: 50%; background: var(--tint); color: var(--navy); font-weight: 800; margin-right: .6rem;}
.strip .num.on {background: linear-gradient(135deg, var(--lavender), var(--blue)); color: #fff;}
.strip .ti {font-weight: 800; color: var(--navy); margin-top: .6rem; font-size: 1.02rem;}
.strip .tx {color: var(--muted); font-size: .9rem; line-height: 1.4;}
.howbox {background: var(--tint-2); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1rem;}
.howbox .h {font-weight: 700; color: var(--navy); margin-bottom: .6rem;}
.howbox .cols {display: flex; gap: 1.5rem; flex-wrap: wrap;}
.howbox .col {flex: 1 1 280px;}
.howbox .col b {display: block; color: var(--navy); margin-bottom: .2rem;}
.howbox .col span {color: var(--muted); font-size: .9rem; line-height: 1.45;}
.steps {display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; color: var(--muted);
        margin: .2rem 0 1.1rem 0; font-size: .92rem;}
.steps .n {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: var(--tint);
           align-items: center; justify-content: center; font-weight: 700; color: var(--navy); margin-right: .35rem;}
.steps .on .n {background: linear-gradient(135deg, var(--lavender), var(--blue)); color: #fff;}
.steps .on {color: var(--navy); font-weight: 700;}
.steps .sep {color: var(--line); margin: 0 .3rem;}
.quiet {color: #8A93A8; font-size: .82rem;}
section[data-testid="stSidebar"] {background: var(--tint-2); border-right: 1px solid var(--line);}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: .5rem .8rem; border-radius: 10px;
        margin-bottom: .15rem; color: var(--navy); font-weight: 600;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: #E6E2FA;}
section[data-testid="stSidebar"] .side-foot {color: var(--muted); font-size: .9rem; margin-top: 2rem;
        border-top: 3px solid var(--blue); padding-top: .6rem; max-width: 150px;}
div[data-testid="stAlert"] {border-radius: 12px;}
div[data-testid="stMetric"] {background: var(--tint); border-radius: 14px; padding: .8rem 1rem;}
</style>
"""

ICONS = {
    "people": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "ribbon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.5 13 17 22l-5-3-5 3 1.5-9"/></svg>',
    "bars": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 20V10"/><path d="M12 20V4"/><path d="M18 20v-6"/></svg>',
    "pill": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/></svg>',
    "calendar": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>',
    "pin": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
    "hospital": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 8v8M8 12h8"/></svg>',
    "target": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></svg>',
    "columns": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "shield": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg>',
    "doc": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></svg>',
}
ROW_ICONS = {"Population": "people", "Condition": "ribbon", "Age": "people", "Sex": "people", "Combination": "columns", "Compare by": "columns",
             "Hospital stays": "hospital", "Hospital": "hospital", "Medication": "pill", "Measurement": "bars",
             "Region": "pin", "Period": "calendar", "Analysis": "bars", "Records requested": "doc", "Purpose": "target"}
LOGO = '<img alt="AIM" src="data:image/png;base64,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"/>'
EU_FLAG = ('<svg width="40" height="27" viewBox="0 0 40 27"><rect width="40" height="27" rx="3" fill="#003399"/>'
           + "".join(f'<circle cx="{20 + 8 * __import__("math").cos(i * 0.5236):.2f}" '
                     f'cy="{13.5 + 8 * __import__("math").sin(i * 0.5236):.2f}" r="1.1" fill="#FFCC00"/>' for i in range(12))
           + "</svg>")


def _go(stage):
    st.session_state["stage"] = stage
    st.session_state["flow_rerun"] = True


def _draft():
    return st.session_state.setdefault("draft", {})


def _escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _card(title, rows, lead="", check=True):
    body = "".join(
        f'<div class="row"><div class="ic">{ICONS.get(ROW_ICONS.get(k, "doc"))}</div>'
        f'<div><div class="k">{_escape(k)}</div><div class="v">{_escape(v)}'
        + (f'<span class="d">{_escape(d)}</span>' if d else "") + "</div></div></div>"
        for k, v, d in rows
    )
    head = (f'<div class="head">{"<span class=chk>" + ICONS["check"] + "</span>" if check else ""}'
            f"<h3>{_escape(title)}</h3></div>")
    lead_html = f'<p class="lead">{_escape(lead)}</p>' if lead else ""
    st.markdown(f'<div class="card">{head}{lead_html}{body}</div>', unsafe_allow_html=True)


def _aggregate_banner():
    st.markdown(f'<div class="banner-ok"><div class="ic">{ICONS["shield"]}</div><div>'
                "<b>Only aggregated information may be released.</b>"
                "<span>No personal or patient-level data will be provided.</span></div></div>",
                unsafe_allow_html=True)


def _title(eyebrow, title, sub=""):
    st.markdown(f'<div class="eyebrow">{_escape(eyebrow)}</div><div class="screen-title">{_escape(title)}</div>'
                + (f'<p class="screen-sub">{_escape(sub)}</p>' if sub else ""), unsafe_allow_html=True)


FLOW_STEPS = [
    ("Describe", "Write your research question in plain language.", "doc"),
    ("Verify", "We check the purpose, feasibility and privacy, and refine the request with you.", "shield"),
    ("Get a safe answer", "You receive aggregated results that meet EHDS rules.", "bars"),
]


def _flow_strip(active=1):
    parts = "".join(
        f'<div class="st"><span class="num{" on" if i == active else ""}">{i}</span>'
        f'<span style="color:#2F56D8;vertical-align:middle">{ICONS[icon]}</span>'
        f'<div class="ti">{_escape(name)}</div><div class="tx">{_escape(text)}</div></div>'
        for i, (name, text, icon) in enumerate(FLOW_STEPS, 1)
    )
    st.markdown(f'<div class="strip">{parts}</div>', unsafe_allow_html=True)


def _header():
    st.markdown(
        f'<div class="hdn-header"><div class="hdn-brand">{LOGO}<div><div class="t">AIM Health Data Navigator</div>'
        '<div class="s">Safe access, brighter research \u2022 Synthetic demonstration \u2022 Simulated EHDS health data access body</div>'
        f'</div></div><div class="hdn-eu">{EU_FLAG}<span>A European<br>Health Data Space<br>initiative</span></div></div>',
        unsafe_allow_html=True)


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
    pairs = _rows_from_population(population)
    if pairs is None:
        rows.append(("Combination", "Contains nested criteria groups", "Edit details is not available for nested groups"))
    elif len(pairs) > 1:
        rows.append(("Combination", _boolean_summary(pairs), "Read top to bottom; OR joins a row with the one above"))
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
            rows.append(("Condition", value, f"Identified using ICD-10 {', '.join(item.get('icd10', []))} codes"))
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
    group_by = criteria.get("group_by", []) if isinstance(criteria, dict) else []
    if group_by:
        names = [DIM_LABELS.get(g.get("dimension"), g.get("dimension")) for g in group_by]
        detail = ""
        for g in group_by:
            if g.get("dimension") == "age_band":
                detail = "Age bands: " + ", ".join(b["label"] for b in g.get("bands", []))
            if g.get("dimension") == "year":
                span = f"{g['from']}\u2013{g['to']}" if g.get("from") and g.get("to") else "2023\u20132025 or the period given"
                detail = (f"Per calendar year ({span}); " + detail) if detail else f"Per calendar year ({span})"
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


def _pending_questions(draft):
    answered = draft.setdefault("clarified", {})
    return [q for q in draft.get("clarifications", []) if q.get("id") not in answered]


def _recheck(draft):
    """Re-evaluate whether the (possibly corrected) criteria can be submitted."""
    from ehds_hdab.nl import Interpretation
    draft["ok"] = Interpretation(draft["criteria"], [], [], list(draft.get("problems", [])), []).ok()


def _apply_clarification(draft, question, action):
    items = draft["criteria"]["population"]["items"]
    if action.get("stop"):
        return "stop"
    if action.get("add"):
        items.append(dict(action["add"]))
    label = action.get("replace_condition_with_stay")
    if label:
        for index, item in enumerate(list(items)):
            if item.get("type") == "condition" and item.get("label") == label:
                stay = {"type": "hospital_stay", "label": label, "icd10": list(item.get("icd10", []))}
                if item.get("onset_start"):
                    stay["period_start"] = item["onset_start"]
                if item.get("onset_end"):
                    stay["period_end"] = item["onset_end"]
                    # "had X in 2025" carries only an end date; a stay "in 2025" means that calendar year.
                    stay.setdefault("period_start", f"{str(item['onset_end'])[:4]}-01-01")
                items[index] = stay
                break
    draft["clarified"][question["id"]] = action
    return "ok"


def _screen_welcome(examples):
    st.markdown('<div class="hero"><div class="eyebrow">Ask health data</div><h1>What would you like to research?</h1>'
                "<p>Use everyday language to describe your research question. We'll help you turn it into a "
                "safe and valid request for the European Health Data Space.</p></div>", unsafe_allow_html=True)
    if "pending_question" in st.session_state:
        st.session_state["research_question"] = st.session_state.pop("pending_question")
    question = st.text_area("Your research question", key="research_question", height=140,
                            placeholder="Show HbA1c trends in patients with type 2 diabetes.",
                            label_visibility="collapsed")
    left, _ = st.columns([1, 3])
    if left.button("Check my request \u2192", type="primary", key="welcome_continue", width="stretch"):
        if not question.strip():
            st.warning("Please describe what you would like to research, or pick an example below.")
        else:
            interpretation = interpret_question(question)
            st.session_state["draft"] = {
                "question": question, "criteria": interpretation.criteria,
                "assumptions": list(interpretation.assumptions),
                "unrecognised": list(interpretation.unrecognised),
                "problems": list(interpretation.problems),
                "clarifications": list(interpretation.clarifications),
                "ok": interpretation.ok(), "source": "question", "clarified": {},
            }
            _go("understand")
    _flow_strip(active=1)
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
    st.markdown(
        f'<div class="howbox"><div class="h">How it works: Guardian checks purpose and privacy \u2022 '
        'Evidence computes the numbers</div><div class="cols">'
        f'<div class="col"><b>Guardian</b><span>Checks your request against the permitted (Art. 53) and prohibited '
        '(Art. 54) purposes, applies the small-cell rule and the disclosure ledger, and ensures only aggregated '
        'information can be released.</span></div>'
        '<div class="col"><b>Evidence</b><span>Calculates every statistic deterministically from the data with '
        'documented clinical definitions, so each result is reproducible and independently verifiable.</span></div>'
        '</div></div>', unsafe_allow_html=True)
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
    _flow_strip(active=2)
    _title("Verify", "Here's how I understood your research question", f"\u201c{draft.get('question', '')}\u201d")
    rows = _interpretation_rows(criteria) if draft.get("ok") else []
    if rows:
        _card("Your request, understood", rows,
              lead="We've identified the key elements of your request. Please confirm they are correct.")
        _aggregate_banner()
    if draft.get("assumptions"):
        st.markdown("**I assumed:**")
        for a in draft["assumptions"]:
            st.markdown(f"- {a}")
    if draft.get("unrecognised"):
        st.warning("I could not match these words to the available data: "
                   + ", ".join(f"\u201c{w}\u201d" for w in draft["unrecognised"])
                   + ". They were not used. The data covers 15 diagnoses, 17 medication groups and 4 laboratory "
                     "tests; use *Edit details* to choose from them.")
    for problem in draft.get("problems", []):
        st.error(problem)
    questions = _pending_questions(draft)
    if questions and not draft.get("problems"):
        st.info("One detail needs clarification before I can prepare the request: " + questions[0]["question"])
    elif not draft.get("ok") and not draft.get("problems"):
        st.error("I could not find a condition, medication, measurement or patient group in your question that the "
                 "data supports. Rephrase it or build the request step by step.")
    st.caption("Interpretation is rule-based (no AI model) and uses the clinical definitions documented for this "
               "prototype. Patient data never leaves this computer.")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    can_continue = not draft.get("problems") and (draft.get("ok") or bool(questions))
    if can_continue and c1.button("Looks good — Continue" if draft.get("ok") else "Answer the question",
                                  type="primary", key="understand_continue", width="stretch"):
        _go("clarify" if (_pending_clarifications(draft) or questions) else "review")
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
    questions = _pending_questions(draft)
    st.markdown("### One quick question" if len(pending) + len(questions) == 1 else "### A few quick questions")
    if not pending and not questions:
        _recheck(draft)
        _go("review" if draft.get("ok") else "understand")
        return
    answers = {}
    for question in questions:
        st.markdown(f"**{question['question']}**")
        labels = [label for label, _ in question["options"]]
        answers[question["id"]] = st.radio(question["id"], labels, key=f"clarify_{question['id']}",
                                           label_visibility="collapsed")
        if question.get("note"):
            st.caption(question["note"])
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
        for question in questions:
            action = dict(question["options"])[answers[question["id"]]]
            if _apply_clarification(draft, question, action) == "stop":
                st.session_state["pending_question"] = draft.get("question", "")
                _go("welcome")
                return
        _recheck(draft)
        leaves, _ = _criterion_leaves(draft["criteria"]["population"]["items"])
        for item in leaves:
            if item.get("type") == "medication" and item.get("label") in choices:
                status = "ever" if choices[item["label"]].startswith("Ever") else "active"
                item["status"] = status
                draft["clarified"][item["label"]] = status
        draft["assumptions"] = [a for a in draft["assumptions"] if not a.startswith("Medication status for")]
        _go("review" if draft.get("ok") else "understand")
    if c2.button("Back", key="clarify_back", width="stretch"):
        _go("understand")


def _screen_review():
    draft = _draft()
    criteria = draft.get("criteria") or {}
    _flow_strip(active=2)
    _title("Verify", "Your research request is ready for review",
           f"\u201c{draft['question']}\u201d" if draft.get("question") else "")
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
    _flow_strip(active=3)
    _title("Safe answer", "Your research results" if approved else "Decision on your request")
    if resp.get("withheld"):
        st.write("Your purpose is permitted, but releasing this particular result could reveal a protected small group "
                 "when combined with statistics released earlier, so it is withheld.")
    else:
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
    parts = [f'<span class="{"on" if i == step else ""}"><span class="n">{i}</span>{name}</span>'
             for i, name in enumerate(GUIDED_STEPS, 1)]
    st.markdown('<div class="steps">' + '<span class="sep">&mdash;</span>'.join(parts) + "</div>",
                unsafe_allow_html=True)


def _population_step(data):
    """Criteria rows, time period and ALL/ANY. Returns (items, match, period_start, period_end, invalid)."""
    years = set()
    for table, column in (("conditions", "onset_date"), ("observations", "date"),
                          ("medications", "start_date"), ("encounters", "start_date")):
        years.update(int(v) for v in data[table][column].dropna().dt.year.unique() if int(v) <= 2025)
    year_options = [0, *sorted(years)]
    st.markdown("**Time period** (optional)")
    y1, y2 = st.columns(2)
    for key in ("builder_from_year", "builder_to_year"):
        if st.session_state.get(key) not in year_options:
            st.session_state[key] = 0
    from_year = _year_value(y1.selectbox("From year", year_options, key="builder_from_year",
                                         format_func=lambda v: "No start year" if not v else str(v)))
    to_year = _year_value(y2.selectbox("To year", year_options, key="builder_to_year",
                                       format_func=lambda v: "No end year" if not v else str(v)))
    st.caption("Lab, medication and hospital-stay records cover 2023–2025; diagnoses go back further.")
    period_start, period_end = _year_bounds(from_year, to_year)
    invalid_period = from_year is not None and to_year is not None and from_year > to_year
    if invalid_period:
        st.error("The start year must not be after the end year.")
    st.markdown("**Criteria**")
    st.caption("Combine rows like a library catalogue search: AND, OR, AND NOT, OR NOT. Rows are read from top "
               "to bottom; OR joins a row with the one above, AND starts a new group.")
    categories = ["Diagnosis", "Medication", "Laboratory measurement", "Age", "Sex", "Region",
                  "Hospital (data holder)", "Hospital stay (main diagnosis)"]
    a1, a2 = st.columns([2, 1])
    category = a1.selectbox("Add a criterion", categories, key="builder_new_category")
    if a2.button("Add criterion", key="builder_add_criterion", width="stretch"):
        row_id = st.session_state.get("builder_next_criterion_id", 0) + 1
        st.session_state["builder_next_criterion_id"] = row_id
        st.session_state.setdefault("builder_criteria", []).append({"id": row_id, "category": category})
    st.session_state.setdefault("builder_criteria", [])
    pairs = []
    rows = list(st.session_state["builder_criteria"])
    for row in rows:
        with st.container(border=True):
            criterion = _builder_criterion_row(row, data, period_start, period_end, first=row is rows[0])
        if criterion is not None:
            pairs.append((st.session_state.get(f"builder_{row['id']}_op", "AND"), criterion))
    if not rows:
        st.caption("No criteria yet. Add at least one, e.g. a diagnosis.")
    elif len(pairs) > 1:
        st.markdown(f"Combination: **{_escape(_boolean_summary(pairs))}**")
    return _combine_rows(pairs), "all", period_start, period_end, invalid_period


def _analysis_step(data, period_start, period_end):
    """Returns (group_by, metrics)."""
    st.markdown("**What would you like to find out?**")
    analysis = st.radio("Analysis", ["Count patients", "Compare groups", "Calculate a mean", "Explore changes over time"],
                        key="guided_analysis", label_visibility="collapsed")
    explanations = {
        "Count patients": "How many patients meet your criteria.",
        "Compare groups": "See how results differ between men and women, age bands, regions or hospitals.",
        "Calculate a mean": "The average of each patient's latest laboratory value (within the period if set).",
        "Explore changes over time": "Results per calendar year, shown as a line graph. Each year counts diagnoses "
                                     "recorded by 31 December, medication status on 31 December, the latest "
                                     "laboratory value within the year and hospital stays within the year.",
    }
    st.caption(explanations[analysis])
    group_by = []
    if analysis == "Explore changes over time":
        all_years = sorted({int(v) for table, column in (("conditions", "onset_date"), ("observations", "date"),
                                                        ("medications", "start_date"), ("encounters", "start_date"))
                            for v in data[table][column].dropna().dt.year.unique() if int(v) <= 2025})
        y1, y2 = st.columns(2)
        for key, default in (("builder_year_from", 2023), ("builder_year_to", 2025)):
            if st.session_state.get(key) not in all_years:
                st.session_state[key] = default if default in all_years else all_years[-1]
        year_from = y1.selectbox("From year", all_years, key="builder_year_from")
        year_to = y2.selectbox("To year", all_years, key="builder_year_to")
        if year_from > year_to:
            st.warning("The first year must not be after the last year.")
            year_from, year_to = year_to, year_from
        group_by.append({"dimension": "year", "from": int(year_from), "to": int(year_to)})
        st.caption("Laboratory, medication and hospital data cover 2023\u20132025; earlier years show diagnoses only. "
                   "Age is as of 31 December 2025 for every year.")
    if analysis in ("Compare groups", "Calculate a mean", "Explore changes over time"):
        choices = {"Sex": "sex", "Age band": "age_band", "Region": "region", "Hospital": "data_holder"}
        selected = st.multiselect("Also compare by" if analysis == "Explore changes over time" else "Compare by",
                                  list(choices), key="builder_group_by")
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
        analysis in ("Compare groups", "Explore changes over time")
        and st.checkbox("Also compare a laboratory value", key="builder_mean_enabled"))
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
        period_start, period_end = _year_bounds(_year_value(st.session_state.get("builder_from_year")),
                                                _year_value(st.session_state.get("builder_to_year")))
        group_by, metrics = _analysis_step(data, period_start, period_end)
        c1, c2 = st.columns([1, 1])
        if c1.button("Next", type="primary", key="guided_next_3", width="stretch"):
            # Recompute the population from the stored rows without rendering them again.
            items = st.session_state.get("guided_items", [])
            criteria = {"population": {"match": "all", "items": items}, "group_by": group_by,
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
    _title("Judges", "Demo scenarios")
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
    _title("Administrative staff", "Staff dashboard")
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
                 "metric": e.get("metric"), "released value": e.get("released") if e.get("released") is not None else "—",
                 "status": e.get("status"),
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
    _title("Approved researchers", "Secure processing environment (SPE)")
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
    _title("Validation", "Evaluation on the 10 challenge requests")
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
st.markdown(STYLE, unsafe_allow_html=True)
_header()
st.sidebar.markdown('<div class="eyebrow">Navigate</div>', unsafe_allow_html=True)
_page = st.sidebar.radio("View", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown('<div class="side-foot">Better health data for a healthier Europe</div>', unsafe_allow_html=True)
st.sidebar.caption("Simulated HDAB for AIM Hackathon Vienna 2026, Track 3. Synthetic data only.")
st.sidebar.caption(PROTOTYPE_DISCLAIMER)
PAGES[_page]()
