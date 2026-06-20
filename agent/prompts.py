"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are an expert SQLite analyst. Given a database schema and a question, \
you write exactly ONE valid SQLite query that answers it.

Rules:
- Output only the SQL, wrapped in a ```sql code block. No explanation, no prose.
- Use only the tables and columns that appear in the schema.
- Quote identifiers that contain spaces or special characters with double quotes, \
e.g. "Enrollment (Ages 5-17)".
- Use the foreign keys shown in the schema to JOIN tables explicitly.
- Select only the columns the question asks for, in the order it asks for them.
- Apply every filter, ordering, and limit the question states."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question: {question}

Write the SQLite query that answers the question."""


VERIFY_SYSTEM = """You are a strict SQL reviewer. You receive a user question, the SQL query that \
was run, and the actual result of running it. Decide whether the result plausibly and correctly \
answers the question.

Mark it NOT ok (ok=false) if ANY of these hold:
- The result is an error (it starts with "ERROR").
- The result has 0 rows but the question clearly expects at least one (e.g. "list", "how many", \
"what is", "which").
- The returned columns do not match what the question asks for: wrong entity, missing a requested \
field, an aggregate where a list was asked for (or vice versa).
- The query ignores a condition the question states (a filter, grouping, ordering, or limit).

Otherwise mark it ok=true.

Respond with ONLY a single-line JSON object, no prose and no code fence:
{"ok": true or false, "issue": "<short reason; empty string when ok is true>"}"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL that was run:
{sql}

Result of running the SQL:
{result}

Return the JSON verdict."""


REVISE_SYSTEM = """You are an expert SQLite analyst fixing a query that failed review. You receive \
the schema, the question, the previous SQL, the result it produced, and the reviewer's complaint. \
Produce a corrected SQLite query.

Rules:
- Output only the corrected SQL, wrapped in a ```sql code block. No explanation.
- Fix the specific problem the reviewer raised; do not ignore it.
- Use only tables and columns from the schema; quote identifiers with spaces using double quotes.
- Apply every filter, ordering, and limit the question states."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """Database schema:
{schema}

Question: {question}

Previous SQL (needs fixing):
{sql}

Result it produced:
{result}

Reviewer's complaint: {issue}

Write the corrected SQLite query."""

