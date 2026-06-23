import json
import os
import re
from flask import Flask, request, jsonify, render_template
import anthropic

app = Flask(__name__)

# Load knowledgebase once at startup
KB_PATH = os.path.join(os.path.dirname(__file__), "opera_tables.json")
with open(KB_PATH) as f:
    TABLES = json.load(f)

# Load golden examples if available
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_examples.json")
GOLDEN_EXAMPLES = []
if os.path.exists(GOLDEN_PATH):
    with open(GOLDEN_PATH) as f:
        GOLDEN_EXAMPLES = json.load(f)

def build_golden_examples_context() -> str:
    """Format golden examples as additional prompt context."""
    if not GOLDEN_EXAMPLES:
        return ""
    lines = ["\n\n--- VERIFIED WORKING OPERA CLOUD QUERY EXAMPLES ---"]
    lines.append("The following are real, production-verified queries. Use them as reference for correct table names, join patterns, column names, and Opera-specific conventions.\n")
    for ex in GOLDEN_EXAMPLES:
        lines.append(f"### Example: {ex['title']}")
        lines.append(f"Purpose: {ex['description']}\n")
        lines.append("Key patterns in this query:")
        for p in ex.get("key_patterns", []):
            lines.append(f"  - {p}")
        lines.append(f"\nSQL:\n{ex['sql']}\n")
    return "\n".join(lines)

# Build a quick-lookup index: table_name -> table object
TABLE_INDEX = {t["table_name"].upper(): t for t in TABLES}

# Prefixes that indicate BI/reporting copies — penalise in scoring so base tables win
SECONDARY_PREFIXES = ("OBI_", "OBITAB_")

# Build a searchable flat list
SEARCH_INDEX = []
for t in TABLES:
    col_names = " ".join(c["name"] for c in t.get("columns", []))
    col_descs = " ".join(c.get("description", "") for c in t.get("columns", []))
    name_upper = t["table_name"].upper()
    is_secondary = name_upper.startswith(SECONDARY_PREFIXES)
    SEARCH_INDEX.append({
        "table_name": t["table_name"],
        "module": t.get("module", ""),
        "description": t.get("description", ""),
        "is_secondary": is_secondary,
        "searchable": f"{t['table_name']} {t.get('module','')} {t.get('description','')} {col_names} {col_descs}".lower()
    })


def find_relevant_tables(query: str, max_tables: int = 12) -> list:
    """Score tables by keyword overlap, penalising BI-layer secondary tables."""
    query_lower = query.lower()
    stopwords = {"the", "a", "an", "for", "with", "and", "or", "of", "in",
                 "to", "from", "all", "get", "show", "me", "give", "list",
                 "find", "return", "where", "that", "have", "has", "is", "are"}
    words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in stopwords and len(w) > 2]

    scores = []
    for entry in SEARCH_INDEX:
        score = sum(entry["searchable"].count(w) for w in words)
        for w in words:
            if w in entry["table_name"].lower():
                score += 5
        # Penalise OBI_/OBITAB_ tables so base tables are preferred when both score similarly
        if entry["is_secondary"] and score > 0:
            score = max(1, score - 10)
        if score > 0:
            scores.append((score, entry["table_name"]))

    scores.sort(reverse=True)

    # Post-filter: if a secondary table made the cut but a base equivalent also scored,
    # drop the secondary one to avoid sending both unnecessarily
    top_names = [name for _, name in scores[:max_tables * 2]]
    filtered = []
    for name in top_names:
        name_upper = name.upper()
        is_sec = name_upper.startswith(SECONDARY_PREFIXES)
        if is_sec:
            # Find the base name by stripping the prefix
            for prefix in SECONDARY_PREFIXES:
                if name_upper.startswith(prefix):
                    base = name_upper[len(prefix):]
                    if base in TABLE_INDEX and base in top_names:
                        continue  # drop secondary — base is already present
            filtered.append(name)
        else:
            filtered.append(name)
        if len(filtered) == max_tables:
            break

    return [TABLE_INDEX[n] for n in filtered if n in TABLE_INDEX]


def extract_tables_from_sql(sql: str) -> list:
    """Extract table names referenced in a SQL query and look them up in the knowledgebase."""
    words = re.findall(r'\b[A-Z_$]{3,}\b', sql.upper())
    found = [TABLE_INDEX[w] for w in set(words) if w in TABLE_INDEX]
    return found


def format_table_context(tables: list) -> str:
    """Format table definitions as compact context for the LLM."""
    lines = []
    for t in tables:
        tag = " [BI-LAYER COPY]" if t["table_name"].upper().startswith(SECONDARY_PREFIXES) else ""
        lines.append(f"\nTABLE: {t['table_name']}{tag} (Module: {t.get('module','')}, Type: {t.get('table_type','')})")
        lines.append(f"Description: {t.get('description','')}")
        lines.append("Columns:")
        for c in t.get("columns", []):
            nullable = "NULL" if c.get("nullable") else "NOT NULL"
            lines.append(f"  - {c['name']} {c['data_type']} {nullable}: {c.get('description','')}")
    return "\n".join(lines)


SYSTEM_PROMPT_BASE = """You are an expert Oracle SQL query builder for Opera Cloud (OPERA 5), a hotel property management system.

You are given a subset of the Opera Cloud data dictionary (relevant tables and their columns).
Your job is to generate correct, production-ready Oracle SQL queries based on the user's request.

Rules:
- Use Oracle SQL syntax (not MySQL or PostgreSQL)
- Use table aliases for readability
- Use bind variables with colon prefix (e.g. :p_resort, :p_from_date) for user-supplied filter values
- Always add a brief comment block at the top of the SQL explaining the query purpose
- After the SQL, provide a clear explanation section titled "## Logic & Joins" that explains:
  - Which tables were used and why
  - How each JOIN works and on what keys
  - Any assumptions made
  - Suggested indexes or performance notes if relevant
- If the request is ambiguous, state your assumption clearly
- If a column does not exist in the provided table definitions, say so explicitly — never invent column names

Opera Cloud Conventions (learned from production queries — follow exactly):
- Multi-environment join keys are DSI and ORGANIZATIONID (exact casing) — always include both in JOINs:
    ON a.ORGANIZATIONID = b.ORGANIZATIONID AND a.DSI = b.DSI
- RESORT is always a WHERE filter: WHERE r.RESORT = :p_resort
- RESERVATION_SUMMARY.EVENT_ID is the same identifier as RESV_NAME_ID in other tables
- RESERVATION_NAME must be joined with NAME_USAGE_TYPE = 'DG' to get the primary (direct) guest
- RESERVATION_NAME joins to RESERVATION_SUMMARY on CONFIRMATION_NO (not RESV_NAME_ID)
- RAHTL_ prefixed tables/views (e.g. RAHTL_RESERVATION_GENERAL_VIEW, rahtl_profile_documents_view) are legitimate production views — use them freely
- RESORT$_ and ALLOTMENT$HEADER use $ in their names — do not alter these names
- NAME_ADDRESS should be deduplicated using ROW_NUMBER() OVER (PARTITION BY name_id, organizationid, dsi ORDER BY address_id DESC) with filters PRIMARY_YN='Y', INACTIVE_DATE IS NULL, NVL(DELETED_FLAG,'N')='N'
- APPLICATION$_USER is joined on APP_USER_ID (not USER_ID) plus DSI and ORGANIZATIONID
- Use REGEXP_REPLACE(col, '[[:cntrl:]|]', '') to clean string columns
- Use TO_CHAR(date_col, 'YYYY-MM-DD') for date formatting
- Use NVL(:p_from_date, TRUNC(SYSDATE)) pattern for optional date bind variables
- ALLOTMENT$HEADER status IN ('O','F') means open/fixed group blocks
- Document ID type/number queries use rahtl_profile_documents_view with primary_yn='Y', inactive_date IS NULL, id_type NOT LIKE '%SUPPO%', ROWNUM <= 1
- RESERVATION_STAT_DAILY is for PAST/historical stays; it contains denormalized guest attributes directly: STAY_ROOMS, NIGHTS, INSERT_DATE, BIRTH_DATE, NATIONALITY, COUNTRY — do NOT try to join NAME or profile tables to get these
- RESERVATION_SUMMARY is for FUTURE/current reservations (departure_date >= TRUNC(SYSDATE) or arrival_date >= TRUNC(SYSDATE))
- prt_property uses datasourceid (not dsi) in its join: ON t.organizationid = p.organizationid AND t.resort = p.resort AND p.datasourceid = t.dsi
- rahtl_profile_all_view fields include: name_type_description, tax1_no, country_code, last, first — use these exact names
- application_parameter_values is filtered by parameter_name (e.g. WHERE parameter_name = 'FISCAL_FOLIO_VALIDATION') to retrieve system config values
- Use CTEs (WITH clause) freely when queries have multiple logical stages — they improve readability and Oracle optimises them well
- rahtl_folio_tax_view provides fiscal/tax line details; join on organizationid, dsi, resort, and folio_no
- RAHTL_reservation_VIEW is the main reservation view with actual_check_in_date, resv_status, name_usage_type, parent_resv_name_id
- name_usage_type = 'DG' is the primary/direct guest; name_usage_type = 'AG' is an accompanying guest — use parent_resv_name_id to link AG back to the main reservation
- RAHTL_NAME_KEYWORDS_VIEW joined twice for Spanish last names: keyword_type='MLASTNAME' and keyword_type='PLASTNAME'
- RAHTL_PROFILE_RELATIONSHIP_VW joined on OWNER_PROFILE_ID + PRIMARY_FLAG='Y' to get relationship type between guests
- ISO_COUNTRY_CODES_EV joined twice (once for nationality, once for country) converting CHAR_CODE2 to CHAR_CODE3 (ISO 3-letter)
- rahtl_exp_mapping_view: mapping_code='MC_GENERIC2' for support document type; MC_GENERIC1 for primary document
- na.name_type = 'D' filters individual guest profiles; 'C' is company
- RAHTL_ views are now fully in the knowledgebase — prefer them for reporting queries

Table Selection Rules:
- Tables tagged [BI-LAYER COPY] (prefixed OBI_ or OBITAB_) are Oracle BI denormalized copies — avoid them unless the user explicitly asks for BI/OBI data. Always prefer the equivalent base table.
- If only a [BI-LAYER COPY] table is available for a concept, use it but note this in the Logic & Joins section.

R&A Multi-Environment Performance Rules (CRITICAL):
These rules are based on documented optimizations that reduced query execution from 2 hours to 2 minutes:
- Always include ORGANIZATIONID AND DSI in every JOIN condition between tables
- Always include RESORT as a filter condition
- Recommend composite indexes on (ORGANIZATIONID, DSI) for all joined tables
- Never omit ORGANIZATIONID or DSI from join conditions — missing these causes full table scans across all environments"""

def build_system_prompt() -> str:
    return SYSTEM_PROMPT_BASE + build_golden_examples_context()

REVIEW_PROMPT = """You are an expert Oracle SQL reviewer and optimiser for Opera Cloud (OPERA 5), a hotel property management system.

You are given an existing SQL query and the relevant Opera Cloud table definitions from the data dictionary.
Your job is to review, correct, and optimise the query.

Table Selection Rules:
- Tables prefixed OBI_ or OBITAB_ are Oracle BI denormalized copies (marked [BI-LAYER COPY]). Flag their use and suggest the base table equivalent unless BI data is explicitly needed.

R&A Multi-Environment Optimisation Rules (CRITICAL — check these first):
These are based on documented real-world optimizations that reduced query execution from 2 hours to 2 minutes:
1. Every JOIN must include BOTH organization_id AND dsi in the ON clause:
   CORRECT:   ON a.organization_id = b.organization_id AND a.dsi = b.dsi
   INCORRECT: ON a.some_key = b.some_key  (missing organization_id/dsi = full table scan across all environments)
2. RESORT must always be used as a filter condition where the table has it:
   AND r.RESORT = :resort
3. Composite indexes on (organization_id, dsi) should be recommended for all joined tables.
4. Missing organization_id or dsi from joins is the single biggest performance issue in Opera Cloud R&A queries.

Provide your response in exactly this structure:

## Issues Found
List all correctness and performance problems. ALWAYS check first:
- Are OBI_/OBITAB_ tables used when base tables should be preferred?
- Are ORGANIZATIONID AND DSI included in every JOIN condition?
- Is RESORT filtered where applicable?
- Any invalid column names, wrong Oracle syntax, cartesian products, or missing filters?
- Is RESERVATION_NAME joined with NAME_USAGE_TYPE = 'DG'?
- Is APPLICATION$_USER joined on APP_USER_ID (not USER_ID)?
If none, say "No issues found."

## Optimised Query
Provide the corrected and optimised SQL. Always include a comment block at the top. Ensure all JOINs include ORGANIZATIONID AND DSI.

## What Changed
Bullet list of every change made and why. If nothing changed, say so.

## Performance Notes
Recommend composite indexes on (ORGANIZATIONID, DSI), partition pruning opportunities, and any other rewrites that improve performance on large multi-environment Opera Cloud datasets."""

REFINE_SYSTEM_PROMPT = """You are an expert Oracle SQL query builder for Opera Cloud (OPERA 5), a hotel property management system.
You are in a follow-up conversation refining a previously generated SQL query based on user feedback.

Rules:
- Use Oracle SQL syntax (not MySQL or PostgreSQL)
- Use table aliases for readability
- Always include ORGANIZATIONID AND DSI in every JOIN condition (exact casing)
- Always include RESORT as a WHERE filter
- Avoid OBI_/OBITAB_ tables unless explicitly requested
- RESERVATION_NAME must always be joined with NAME_USAGE_TYPE = 'DG'
- APPLICATION$_USER must be joined on APP_USER_ID (not USER_ID)
- Use REGEXP_REPLACE(col, '[[:cntrl:]|]', '') to clean string columns
- After the updated SQL, provide a brief "## What Changed" section explaining the modifications
- After that, provide an updated "## Logic & Joins" section"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def build_query():
    data = request.get_json()
    user_request = data.get("request", "").strip()
    if not user_request:
        return jsonify({"error": "No request provided"}), 400

    relevant_tables = find_relevant_tables(user_request)
    if not relevant_tables:
        return jsonify({"error": "Could not find relevant tables for your request. Try different keywords."}), 400

    table_context = format_table_context(relevant_tables)
    table_names_used = [t["table_name"] for t in relevant_tables]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=build_system_prompt(),
        messages=[
            {
                "role": "user",
                "content": f"""Here are the relevant Opera Cloud tables from the data dictionary:

{table_context}

---

User request: {user_request}

Please generate the Oracle SQL query and explain the logic."""
            }
        ]
    )

    response_text = message.content[0].text

    sql_part = ""
    logic_part = ""
    if "## Logic & Joins" in response_text:
        parts = response_text.split("## Logic & Joins", 1)
        sql_part = parts[0].strip()
        logic_part = "## Logic & Joins" + parts[1]
    else:
        sql_part = response_text

    return jsonify({
        "sql": sql_part,
        "logic": logic_part,
        "tables_searched": table_names_used,
        "table_context": table_context
    })


@app.route("/refine", methods=["POST"])
def refine_query():
    data = request.get_json()
    original_sql = data.get("sql", "").strip()
    original_logic = data.get("logic", "").strip()
    table_context = data.get("table_context", "").strip()
    follow_up = data.get("follow_up", "").strip()
    history = data.get("history", [])

    if not original_sql or not follow_up:
        return jsonify({"error": "Missing SQL or follow-up request"}), 400

    messages = [
        {
            "role": "user",
            "content": f"""Here are the relevant Opera Cloud tables from the data dictionary:

{table_context}

---

Original SQL query:

{original_sql}

{original_logic}"""
        },
        {
            "role": "assistant",
            "content": f"{original_sql}\n\n{original_logic}"
        }
    ]

    # Append any prior refinement turns
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": follow_up})

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=REFINE_SYSTEM_PROMPT,
        messages=messages
    )

    response_text = message.content[0].text

    sql_part = ""
    logic_part = ""
    if "## Logic & Joins" in response_text:
        parts = response_text.split("## Logic & Joins", 1)
        sql_part = parts[0].strip()
        logic_part = "## Logic & Joins" + parts[1]
    else:
        sql_part = response_text

    return jsonify({
        "sql": sql_part,
        "logic": logic_part
    })


@app.route("/review", methods=["POST"])
def review_query():
    data = request.get_json()
    sql = data.get("sql", "").strip()
    if not sql:
        return jsonify({"error": "No SQL provided"}), 400

    tables = extract_tables_from_sql(sql)
    if not tables:
        return jsonify({"error": "Could not identify any known Opera Cloud tables in the SQL. Check table names are correct."}), 400

    table_context = format_table_context(tables)
    table_names_used = [t["table_name"] for t in tables]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=REVIEW_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Here are the Opera Cloud table definitions for the tables referenced in this query:

{table_context}

---

SQL to review:

{sql}

Please review, correct, and optimise this query."""
            }
        ]
    )

    response_text = message.content[0].text

    return jsonify({
        "review": response_text,
        "tables_found": table_names_used
    })


@app.route("/tables", methods=["GET"])
def list_tables():
    """Search tables by keyword."""
    q = request.args.get("q", "").lower()
    results = [
        {"table_name": t["table_name"], "module": t.get("module", ""), "description": t.get("description", "")}
        for t in TABLES
        if q in t["table_name"].lower() or q in t.get("description", "").lower() or q in t.get("module", "").lower()
    ]
    return jsonify(results[:50])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
