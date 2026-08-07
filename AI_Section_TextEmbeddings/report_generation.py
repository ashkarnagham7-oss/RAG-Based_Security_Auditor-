from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Configuration
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent

PROCESSED_PROJECT_NAME = "flask_webgoat"

PROCESSED_PROJECT_DIRECTORY = (
    PROJECT_ROOT
    / "processed"
    / PROCESSED_PROJECT_NAME
)

FINDINGS_FILE = (
    PROCESSED_PROJECT_DIRECTORY
    / "vulnerability_findings.jsonl"
)

REASONING_SUMMARY_FILE = (
    PROCESSED_PROJECT_DIRECTORY
    / "reasoning_summary.json"
)

REASONING_ERRORS_FILE = (
    PROCESSED_PROJECT_DIRECTORY
    / "reasoning_errors.jsonl"
)

HTML_REPORT_FILE = (
    PROCESSED_PROJECT_DIRECTORY
    / "security_report.html"
)

JSON_REPORT_FILE = (
    PROCESSED_PROJECT_DIRECTORY
    / "security_report.json"
)


# File helpers
def require_file(
    file_path: Path,
    description: str,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n"
            f"{file_path.resolve()}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"{description} is not a file:\n"
            f"{file_path.resolve()}"
        )


def read_json_file(
    file_path: Path,
) -> dict[str, Any]:
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{file_path.name} must contain one JSON object."
        )

    return data


def read_jsonl_file(
    file_path: Path,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    if not file_path.exists():
        if allow_missing:
            return []

        raise FileNotFoundError(
            f"File was not found:\n"
            f"{file_path.resolve()}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {file_path.name}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of "
                    f"{file_path.name} is not a JSON object."
                )

            records.append(record)

    return records


def write_text_file(
    content: str,
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(content)


def write_json_file(
    data: dict[str, Any],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# Formatting helpers

def text_value(
    value: Any,
    default: str = "Not available",
) -> str:
    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


def escape(
    value: Any,
    default: str = "Not available",
) -> str:
    return html.escape(
        text_value(value, default),
        quote=True,
    )


def concise_reason(
    value: Any,
    maximum_length: int = 260,
) -> str:
    text = text_value(value)

    sentence_end = text.find(". ")

    if sentence_end != -1:
        text = text[:sentence_end + 1]

    if len(text) <= maximum_length:
        return text

    return text[:maximum_length].rstrip() + "..."


def format_location(
    finding: dict[str, Any],
) -> str:
    file_path = text_value(
        finding.get("file_path")
    )

    start_line = finding.get("start_line")
    end_line = finding.get("end_line")

    return (
        f"{file_path}:{start_line}-{end_line}"
    )


def format_lines(
    lines: Any,
) -> str:
    if not isinstance(lines, list) or not lines:
        return "Not specified"

    normalized: list[int] = []

    for line in lines:
        try:
            normalized.append(int(line))
        except (TypeError, ValueError):
            continue

    if not normalized:
        return "Not specified"

    normalized = sorted(set(normalized))

    return ", ".join(
        str(line)
        for line in normalized
    )


def confidence_to_severity(
    finding: dict[str, Any],
) -> str:

    level = str(
        finding.get("confidence_level", "")
    ).strip().lower()

    if level == "high":
        return "HIGH"

    if level == "medium":
        return "MEDIUM"

    return "LOW"


def severity_css_class(
    severity: str,
) -> str:
    normalized = severity.upper()

    if normalized == "CRITICAL":
        return "severity-critical"

    if normalized == "HIGH":
        return "severity-high"

    if normalized == "MEDIUM":
        return "severity-medium"

    return "severity-low"


def extract_required_context(
    finding: dict[str, Any],
) -> str:
    recommendation = finding.get(
        "recommended_fix"
    )

    if recommendation:
        return text_value(recommendation)

    data_flow = finding.get("data_flow")

    if isinstance(data_flow, dict):
        missing = data_flow.get(
            "missing_protection"
        )

        if missing:
            return text_value(missing)

    return (
        "Review the function callers, input origin, "
        "surrounding access-control logic, and external "
        "dependencies."
    )


# ============================================================
# Report preparation
# ============================================================

def group_findings(
    findings: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        "vulnerable": [],
        "uncertain": [],
        "not_vulnerable": [],
    }

    for finding in findings:
        verdict = str(
            finding.get("verdict", "")
        ).strip().lower()

        if verdict in grouped:
            grouped[verdict].append(finding)

    return grouped


def simplify_vulnerability(
    finding: dict[str, Any],
) -> dict[str, Any]:
    severity = (
        finding.get("severity")
        or confidence_to_severity(finding)
    )

    return {
        "function_name": finding.get(
            "function_name"
        ),
        "class_name": finding.get(
            "class_name"
        ),
        "file_path": finding.get(
            "file_path"
        ),
        "start_line": finding.get(
            "start_line"
        ),
        "end_line": finding.get(
            "end_line"
        ),
        "vulnerable_lines": finding.get(
            "vulnerable_lines",
            [],
        ),
        "cwe": finding.get("cwe"),
        "cwe_name": finding.get(
            "cwe_name"
        ),
        "severity": severity,
        "confidence_score": finding.get(
            "confidence_score"
        ),
        "confidence_level": finding.get(
            "confidence_level"
        ),
        "description": finding.get(
            "reason"
        ),
        "vulnerable_code": (
            finding.get(
                "vulnerable_code_excerpt"
            )
            or finding.get("source_code")
        ),
        "fix_suggestion": finding.get(
            "recommended_fix"
        ),
    }


def simplify_uncertain(
    finding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "function_name": finding.get(
            "function_name"
        ),
        "class_name": finding.get(
            "class_name"
        ),
        "file_path": finding.get(
            "file_path"
        ),
        "start_line": finding.get(
            "start_line"
        ),
        "end_line": finding.get(
            "end_line"
        ),
        "reason": concise_reason(
            finding.get("reason"),
            maximum_length=420,
        ),
        "required_context": (
            extract_required_context(
                finding
            )
        ),
    }


def simplify_safe(
    finding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "function_name": finding.get(
            "function_name"
        ),
        "class_name": finding.get(
            "class_name"
        ),
        "file_path": finding.get(
            "file_path"
        ),
        "start_line": finding.get(
            "start_line"
        ),
        "end_line": finding.get(
            "end_line"
        ),
        "reason": concise_reason(
            finding.get("reason"),
            maximum_length=220,
        ),
    }


def build_report_data(
    findings: list[dict[str, Any]],
    reasoning_summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_findings(findings)

    vulnerabilities = [
        simplify_vulnerability(finding)
        for finding in grouped["vulnerable"]
    ]

    uncertain = [
        simplify_uncertain(finding)
        for finding in grouped["uncertain"]
    ]

    safe = [
        simplify_safe(finding)
        for finding in grouped["not_vulnerable"]
    ]

    return {
        "metadata": {
            "project": PROCESSED_PROJECT_NAME,
            "generated_at": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            "reasoning_model": (
                reasoning_summary.get(
                    "reasoning_model"
                )
            ),
        },

        "summary": {
            "functions_analyzed": len(findings),
            "vulnerable": len(vulnerabilities),
            "uncertain": len(uncertain),
            "not_vulnerable": len(safe),
            "analysis_failures": len(errors),
        },

        "vulnerabilities": vulnerabilities,
        "uncertain_findings": uncertain,
        "not_vulnerable_findings": safe,
        "analysis_errors": errors,
    }


# HTML sections

def build_summary_cards(
    summary: dict[str, Any],
) -> str:
    return f"""
    <div class="summary-grid">
        <div class="summary-card summary-total">
            <span class="summary-number">
                {summary["functions_analyzed"]}
            </span>
            <span class="summary-label">
                Functions analysed
            </span>
        </div>

        <div class="summary-card summary-vulnerable">
            <span class="summary-number">
                {summary["vulnerable"]}
            </span>
            <span class="summary-label">
                Vulnerable
            </span>
        </div>

        <div class="summary-card summary-uncertain">
            <span class="summary-number">
                {summary["uncertain"]}
            </span>
            <span class="summary-label">
                Uncertain
            </span>
        </div>

        <div class="summary-card summary-safe">
            <span class="summary-number">
                {summary["not_vulnerable"]}
            </span>
            <span class="summary-label">
                Not vulnerable
            </span>
        </div>
    </div>
    """


def build_vulnerability_cards(
    vulnerabilities: list[dict[str, Any]],
) -> str:
    if not vulnerabilities:
        return """
        <p class="empty-message">
            No confirmed vulnerabilities were identified.
        </p>
        """

    cards: list[str] = []

    for index, finding in enumerate(
        vulnerabilities,
        start=1,
    ):
        severity = text_value(
            finding.get("severity"),
            default="UNKNOWN",
        ).upper()

        css_class = severity_css_class(
            severity
        )

        function_name = escape(
            finding.get("function_name")
        )

        location = escape(
            f"{finding.get('file_path')}:"
            f"{finding.get('start_line')}-"
            f"{finding.get('end_line')}"
        )

        cwe = escape(
            finding.get("cwe"),
            default="Unclassified",
        )

        cwe_name = finding.get("cwe_name")

        cwe_display = cwe

        if cwe_name:
            cwe_display += (
                f" — {escape(cwe_name)}"
            )

        description = escape(
            finding.get("description")
        )

        vulnerable_lines = escape(
            format_lines(
                finding.get(
                    "vulnerable_lines"
                )
            )
        )

        fix_suggestion = escape(
            finding.get("fix_suggestion")
        )

        confidence_score = finding.get(
            "confidence_score"
        )

        confidence_level = escape(
            finding.get(
                "confidence_level"
            ),
            default="unknown",
        )

        try:
            confidence_display = (
                f"{float(confidence_score):.2f}"
            )
        except (TypeError, ValueError):
            confidence_display = (
                "Not available"
            )

        code = html.escape(
            text_value(
                finding.get(
                    "vulnerable_code"
                ),
                default="",
            )
        )

        code_section = ""

        if code:
            code_section = f"""
            <details class="code-details">
                <summary>View vulnerable code</summary>
                <pre><code>{code}</code></pre>
            </details>
            """

        cards.append(
            f"""
            <article class="finding-card {css_class}">
                <div class="finding-header">
                    <div>
                        <h3>
                            {index}. {function_name}
                        </h3>
                        <span class="function-location">
                            {location}
                        </span>
                    </div>

                    <span class="severity-badge">
                        {severity}
                    </span>
                </div>

                <div class="finding-meta">
                    <span>
                        <strong>CWE:</strong>
                        {cwe_display}
                    </span>

                    <span>
                        <strong>Confidence:</strong>
                        {confidence_display}
                        ({confidence_level})
                    </span>
                </div>

                <p>
                    <strong>Description:</strong>
                    {description}
                </p>

                <p>
                    <strong>Reported lines:</strong>
                    {vulnerable_lines}
                </p>

                <p>
                    <strong>Fix suggestion:</strong>
                    {fix_suggestion}
                </p>

                {code_section}
            </article>
            """
        )

    return "\n".join(cards)


def build_uncertain_cards(
    uncertain_findings: list[dict[str, Any]],
) -> str:
    if not uncertain_findings:
        return """
        <p class="empty-message">
            No functions were classified as uncertain.
        </p>
        """

    cards: list[str] = []

    for index, finding in enumerate(
        uncertain_findings,
        start=1,
    ):
        function_name = escape(
            finding.get("function_name")
        )

        location = escape(
            f"{finding.get('file_path')}:"
            f"{finding.get('start_line')}-"
            f"{finding.get('end_line')}"
        )

        reason = escape(
            finding.get("reason")
        )

        required_context = escape(
            finding.get(
                "required_context"
            )
        )

        cards.append(
            f"""
            <article class="finding-card uncertain-card">
                <div class="finding-header">
                    <div>
                        <h3>
                            {index}. {function_name}
                        </h3>
                        <span class="function-location">
                            {location}
                        </span>
                    </div>

                    <span class="status-badge uncertain-badge">
                        UNCERTAIN
                    </span>
                </div>

                <p>
                    <strong>Why uncertain:</strong>
                    {reason}
                </p>

                <p>
                    <strong>Additional context needed:</strong>
                    {required_context}
                </p>
            </article>
            """
        )

    return "\n".join(cards)


def build_safe_table(
    safe_findings: list[dict[str, Any]],
) -> str:
    if not safe_findings:
        return """
        <p class="empty-message">
            No functions were classified as not vulnerable.
        </p>
        """

    rows: list[str] = []

    for finding in safe_findings:
        function_name = escape(
            finding.get("function_name")
        )

        location = escape(
            f"{finding.get('file_path')}:"
            f"{finding.get('start_line')}-"
            f"{finding.get('end_line')}"
        )

        reason = escape(
            finding.get("reason")
        )

        rows.append(
            f"""
            <tr>
                <td>
                    <code>{function_name}</code>
                </td>
                <td>
                    <code>{location}</code>
                </td>
                <td>
                    {reason}
                </td>
            </tr>
            """
        )

    return f"""
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Function</th>
                    <th>Location</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_errors_section(
    errors: list[dict[str, Any]],
) -> str:
    if not errors:
        return ""

    rows: list[str] = []

    for error in errors:
        function_name = escape(
            error.get("function_name")
        )

        location = escape(
            f"{error.get('file_path')}:"
            f"{error.get('start_line')}-"
            f"{error.get('end_line')}"
        )

        message = escape(
            error.get("error")
        )

        rows.append(
            f"""
            <tr>
                <td>{function_name}</td>
                <td><code>{location}</code></td>
                <td>{message}</td>
            </tr>
            """
        )

    return f"""
    <section>
        <h2>Analysis Errors</h2>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Function</th>
                        <th>Location</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
    </section>
    """


# HTML report

def build_html_report(
    report_data: dict[str, Any],
) -> str:
    metadata = report_data["metadata"]
    summary = report_data["summary"]

    generated_at = datetime.fromisoformat(
        metadata["generated_at"]
    ).strftime(
        "%Y-%m-%d %H:%M"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Security Audit Report</title>

<style>
    :root {{
        --red: #c0392b;
        --red-soft: #fbe9e7;
        --orange: #f39c12;
        --orange-soft: #fff6df;
        --green: #27864a;
        --green-soft: #eaf7ee;
        --blue: #3498db;
        --grey-50: #fafafa;
        --grey-100: #f3f4f6;
        --grey-200: #e5e7eb;
        --grey-500: #6b7280;
        --grey-800: #1f2937;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: var(--grey-50);
        color: var(--grey-800);
        font-family:
            Arial,
            Helvetica,
            sans-serif;
        line-height: 1.55;
    }}

    .report-container {{
        width: min(1180px, calc(100% - 40px));
        margin: 35px auto 70px;
    }}

    .report-header {{
        background: white;
        border: 1px solid var(--grey-200);
        border-radius: 10px;
        padding: 28px 30px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    }}

    h1 {{
        margin: 0 0 12px;
        color: var(--red);
        font-size: 38px;
    }}

    .report-meta {{
        margin: 4px 0;
        color: var(--grey-500);
    }}

    section {{
        margin-top: 34px;
    }}

    h2 {{
        margin-bottom: 16px;
        padding-bottom: 9px;
        border-bottom: 1px solid var(--grey-200);
        font-size: 25px;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns:
            repeat(4, minmax(150px, 1fr));
        gap: 14px;
        margin-top: 22px;
    }}

    .summary-card {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 105px;
        border-radius: 8px;
        padding: 16px;
        background: white;
        border: 1px solid var(--grey-200);
    }}

    .summary-number {{
        font-size: 32px;
        font-weight: bold;
    }}

    .summary-label {{
        margin-top: 5px;
        color: var(--grey-500);
        text-align: center;
    }}

    .summary-vulnerable {{
        border-top: 5px solid var(--red);
    }}

    .summary-uncertain {{
        border-top: 5px solid var(--orange);
    }}

    .summary-safe {{
        border-top: 5px solid var(--green);
    }}

    .summary-total {{
        border-top: 5px solid var(--blue);
    }}

    .finding-card {{
        background: white;
        border: 1px solid var(--grey-200);
        border-left-width: 6px;
        border-radius: 8px;
        padding: 22px 24px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.035);
    }}

    .severity-critical,
    .severity-high {{
        border-left-color: var(--red);
    }}

    .severity-medium {{
        border-left-color: var(--orange);
    }}

    .severity-low {{
        border-left-color: var(--blue);
    }}

    .uncertain-card {{
        border-left-color: var(--orange);
        background: #fffdf8;
    }}

    .finding-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 20px;
    }}

    .finding-header h3 {{
        margin: 0 0 8px;
        font-size: 24px;
    }}

    .function-location {{
        display: inline-block;
        padding: 4px 8px;
        background: var(--grey-100);
        border-radius: 4px;
        font-family:
            Consolas,
            "Courier New",
            monospace;
        word-break: break-all;
    }}

    .severity-badge,
    .status-badge {{
        flex-shrink: 0;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 0.04em;
    }}

    .severity-badge {{
        color: white;
        background: var(--red);
    }}

    .uncertain-badge {{
        color: #7a4a00;
        background: #ffe1a3;
    }}

    .finding-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 18px;
        padding: 11px 13px;
        background: var(--grey-100);
        border-radius: 6px;
    }}

    .code-details {{
        margin-top: 18px;
    }}

    .code-details summary {{
        cursor: pointer;
        font-weight: bold;
    }}

    pre {{
        margin-top: 12px;
        padding: 16px;
        overflow-x: auto;
        border-radius: 6px;
        background: #171717;
        color: #f5f5f5;
        line-height: 1.45;
    }}

    code {{
        font-family:
            Consolas,
            "Courier New",
            monospace;
    }}

    .table-wrapper {{
        overflow-x: auto;
        background: white;
        border: 1px solid var(--grey-200);
        border-radius: 8px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th,
    td {{
        padding: 13px 15px;
        border-bottom: 1px solid var(--grey-200);
        text-align: left;
        vertical-align: top;
    }}

    th {{
        background: var(--grey-100);
    }}

    tbody tr:last-child td {{
        border-bottom: none;
    }}

    .empty-message {{
        padding: 18px;
        background: white;
        border: 1px solid var(--grey-200);
        border-radius: 8px;
    }}

    .report-note {{
        margin-top: 34px;
        padding: 16px 18px;
        border-radius: 8px;
        background: #eef5fb;
        color: #27445c;
    }}

    @media (max-width: 800px) {{
        .summary-grid {{
            grid-template-columns:
                repeat(2, minmax(140px, 1fr));
        }}

        .finding-header {{
            flex-direction: column;
        }}
    }}

    @media print {{
        body {{
            background: white;
        }}

        .report-container {{
            width: 100%;
            margin: 0;
        }}

        .report-header,
        .finding-card,
        .table-wrapper {{
            box-shadow: none;
        }}

        .code-details {{
            display: none;
        }}
    }}
</style>
</head>

<body>
<main class="report-container">

    <header class="report-header">
        <h1>Security Audit Report</h1>

        <p class="report-meta">
            <strong>Project:</strong>
            {escape(metadata.get("project"))}
        </p>

        <p class="report-meta">
            <strong>Generated on:</strong>
            {escape(generated_at)}
        </p>

        <p class="report-meta">
            <strong>Reasoning model:</strong>
            {escape(metadata.get("reasoning_model"))}
        </p>

        {build_summary_cards(summary)}
    </header>

    <section>
        <h2>Confirmed Vulnerabilities</h2>

        {build_vulnerability_cards(
            report_data["vulnerabilities"]
        )}
    </section>

    <section>
        <h2>Uncertain Findings</h2>

        {build_uncertain_cards(
            report_data["uncertain_findings"]
        )}
    </section>

    <section>
        <h2>Not Vulnerable</h2>

        {build_safe_table(
            report_data["not_vulnerable_findings"]
        )}
    </section>

    {build_errors_section(
        report_data["analysis_errors"]
    )}

    <div class="report-note">
        <strong>Interpretation note:</strong>
        Confirmed vulnerabilities contain a concrete weakness
        identified in the submitted function. Uncertain findings
        require additional caller, input, dependency, or security
        context before a reliable decision can be made.
        Reported lines may include the function declaration as
        a function-level location marker.
    </div>

</main>
</body>
</html>
"""


# Main
def main() -> None:
    print("=" * 70)
    print("SECURITY REPORT GENERATION")
    print("=" * 70)

    print(f"Project: {PROCESSED_PROJECT_NAME}")

    require_file(
        FINDINGS_FILE,
        "vulnerability_findings.jsonl",
    )

    require_file(
        REASONING_SUMMARY_FILE,
        "reasoning_summary.json",
    )

    findings = read_jsonl_file(
        FINDINGS_FILE
    )

    reasoning_summary = read_json_file(
        REASONING_SUMMARY_FILE
    )

    errors = read_jsonl_file(
        REASONING_ERRORS_FILE,
        allow_missing=True,
    )

    report_data = build_report_data(
        findings=findings,
        reasoning_summary=reasoning_summary,
        errors=errors,
    )

    html_report = build_html_report(
        report_data
    )

    write_text_file(
        html_report,
        HTML_REPORT_FILE,
    )

    write_json_file(
        report_data,
        JSON_REPORT_FILE,
    )

    summary = report_data["summary"]

    print("\n" + "=" * 70)
    print("SECURITY REPORT GENERATED")
    print("=" * 70)

    print(
        f"Functions analysed: "
        f"{summary['functions_analyzed']}"
    )

    print(
        f"Vulnerable: "
        f"{summary['vulnerable']}"
    )

    print(
        f"Uncertain: "
        f"{summary['uncertain']}"
    )

    print(
        f"Not vulnerable: "
        f"{summary['not_vulnerable']}"
    )

    print(
        f"HTML report: "
        f"{HTML_REPORT_FILE.resolve()}"
    )

    print(
        f"JSON report: "
        f"{JSON_REPORT_FILE.resolve()}"
    )
    import webbrowser
    webbrowser.open(
    HTML_REPORT_FILE.resolve().as_uri()
    )

if __name__ == "__main__":
    main()
