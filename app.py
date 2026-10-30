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
