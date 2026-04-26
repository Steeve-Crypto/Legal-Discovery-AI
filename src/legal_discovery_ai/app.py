from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st

from legal_discovery_ai.crew import PROJECT_ROOT, ensure_supported_python, run_crew


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file: Any) -> Path:
    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def normalize_result(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, indent=2, ensure_ascii=True)
    return str(result)


def parse_result_payload(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"case_summary": result}
    return {"case_summary": str(result)}


def as_bullets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    text = str(value).strip()
    return [text] if text else []


def render_list_section(title: str, value: Any) -> None:
    st.markdown(f"### {title}")
    items = as_bullets(value)
    if not items:
        st.write("No items provided.")
        return
    for item in items:
        st.markdown(f"- {item}")


def render_attorney_report(result_payload: dict[str, Any]) -> None:
    st.subheader("Attorney-Facing Report")
    summary = result_payload.get("case_summary", "No case summary available.")
    st.markdown("### Case Summary")
    st.write(summary)

    left_col, right_col = st.columns(2)
    with left_col:
        render_list_section("Timeline", result_payload.get("timeline"))
        render_list_section("Parties", result_payload.get("parties"))
        render_list_section("Key Issues", result_payload.get("key_issues"))
    with right_col:
        render_list_section("Critical Risks", result_payload.get("critical_risks"))
        render_list_section("Missing Elements", result_payload.get("missing_elements"))
        render_list_section("Recommended Next Actions", result_payload.get("next_actions"))


def build_report_markdown(result_payload: dict[str, Any]) -> str:
    def section(title: str, value: Any) -> str:
        items = as_bullets(value)
        if not items:
            return f"## {title}\n- No items provided.\n"
        lines = "\n".join(f"- {item}" for item in items)
        return f"## {title}\n{lines}\n"

    parts = [
        "# Legal Discovery Case Brief",
        "",
        "## Case Summary",
        str(result_payload.get("case_summary", "No case summary available.")),
        "",
        section("Timeline", result_payload.get("timeline")),
        section("Parties", result_payload.get("parties")),
        section("Key Issues", result_payload.get("key_issues")),
        section("Critical Risks", result_payload.get("critical_risks")),
        section("Missing Elements", result_payload.get("missing_elements")),
        section("Next Actions", result_payload.get("next_actions")),
    ]
    return "\n".join(parts)


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    max_chars = 100

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            y -= 12
        else:
            clean = line.replace("## ", "").replace("# ", "")
            while len(clean) > max_chars:
                pdf.drawString(50, y, clean[:max_chars])
                clean = clean[max_chars:]
                y -= 14
                if y < 50:
                    pdf.showPage()
                    y = height - 50
            pdf.drawString(50, y, clean)
            y -= 14

        if y < 50:
            pdf.showPage()
            y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer.read()


def render_status_panel(progress_slot: Any, statuses: dict[str, str], pct: int) -> None:
    progress_slot.progress(pct / 100, text=f"Estimated progress: {pct}%")
    st.markdown("### Agent Status")
    for agent_name, status in statuses.items():
        icon = "🟡" if status == "Running" else "🟢" if status == "Completed" else "⚪"
        st.write(f"{icon} {agent_name}: {status}")


def main() -> None:
    try:
        ensure_supported_python()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.set_page_config(page_title="Legal Discovery AI", layout="wide")
    st.title("Legal Discovery AI")
    st.caption(
        "Upload a legal PDF to extract facts, scan risks, and generate a structured case brief."
    )

    with st.sidebar:
        st.header("Configuration")
        st.write(
            "Ensure `.env` has `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or "
            "`GEMINI_API_KEY`."
        )
        supplemental_text = st.text_area(
            "Optional supplemental text",
            value="",
            help="Use for extra notes not present in the PDF.",
        )
        run_btn = st.button("Run Analysis", type="primary")

    uploaded_pdf = st.file_uploader("Upload legal PDF", type=["pdf"])

    if run_btn:
        if uploaded_pdf is None:
            st.error("Please upload a PDF before running analysis.")
            return

        document_path = save_uploaded_pdf(uploaded_pdf)
        st.info(f"Running crew on `{document_path}`...")

        statuses = {
            "Document Parser": "Queued",
            "Risk Scanner": "Queued",
            "Case Brief Writer": "Queued",
        }
        progress_slot = st.empty()
        status_panel_slot = st.container()

        result_box: dict[str, Any] = {"value": None, "error": None}

        def worker() -> None:
            try:
                result_box["value"] = run_crew(
                    document_path=str(document_path),
                    document_text=supplemental_text or None,
                )
            except Exception as exc:  # pragma: no cover
                result_box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        phase_order = ["Document Parser", "Risk Scanner", "Case Brief Writer"]
        phase_idx = 0
        start_time = time.time()

        while thread.is_alive():
            elapsed = time.time() - start_time
            estimated_phase = min(int(elapsed // 10), len(phase_order) - 1)
            phase_idx = max(phase_idx, estimated_phase)

            for idx, agent_name in enumerate(phase_order):
                if idx < phase_idx:
                    statuses[agent_name] = "Completed"
                elif idx == phase_idx:
                    statuses[agent_name] = "Running"
                else:
                    statuses[agent_name] = "Queued"

            pct = min(90, int((elapsed / 30) * 100))
            with status_panel_slot:
                render_status_panel(progress_slot, statuses, max(5, pct))
            time.sleep(0.5)

        thread.join()

        if result_box["error"] is not None:
            st.exception(result_box["error"])
            return

        for agent_name in phase_order:
            statuses[agent_name] = "Completed"
        with status_panel_slot:
            render_status_panel(progress_slot, statuses, 100)

        result = result_box["value"]
        result_text = normalize_result(result)
        result_payload = parse_result_payload(result)

        st.success("Analysis complete.")
        render_attorney_report(result_payload)

        st.subheader("Export Report")
        report_markdown = build_report_markdown(result_payload)
        report_pdf = markdown_to_pdf_bytes(report_markdown)
        export_format = st.radio(
            "Choose export format",
            ["Markdown", "PDF", "JSON"],
            horizontal=True,
        )
        if export_format == "Markdown":
            st.download_button(
                label="One-Click Export",
                data=report_markdown,
                file_name="case_brief_report.md",
                mime="text/markdown",
            )
        elif export_format == "PDF":
            st.download_button(
                label="One-Click Export",
                data=report_pdf,
                file_name="case_brief_report.pdf",
                mime="application/pdf",
            )
        else:
            st.download_button(
                label="One-Click Export",
                data=result_text,
                file_name="case_brief_output.json",
                mime="application/json",
            )

        with st.expander("Raw JSON Output"):
            st.code(result_text, language="json")
        st.download_button(
            label="Download output JSON",
            data=result_text,
            file_name="case_brief_output.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
