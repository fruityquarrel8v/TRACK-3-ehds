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
