from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task

try:
    from crewai import LLM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CrewAI LLM class is unavailable. Upgrade crewai to a recent version."
    ) from exc

try:
    from crewai_tools import (
        DirectorySearchTool,
        FileReadTool,
        PDFSearchTool,
        RagTool,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "crewai_tools is unavailable. Install dependencies via requirements.txt."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAST_CASES_DIR = PROJECT_ROOT / "data" / "past_cases"


def ensure_supported_python() -> None:
    """Fail fast on unsupported Python runtimes."""
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Python 3.14 is not yet supported by CrewAI dependencies on Windows. "
            "Use Python 3.11 or 3.12 (recommended) and recreate your virtual environment."
        )


def build_llm() -> LLM:
    """Build an LLM preferring Anthropic, then OpenAI, then Gemini."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = resolve_gemini_model(
        os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), gemini_key
    )

    if anthropic_key:
        return LLM(
            model="claude-3-5-sonnet-20241022",
            api_key=anthropic_key,
            temperature=0.1,
        )

    if openai_key:
        return LLM(
            model="gpt-4o",
            api_key=openai_key,
            temperature=0.1,
        )

    if gemini_key:
        return LLM(
            model=gemini_model,
            api_key=gemini_key,
            temperature=0.1,
        )

    raise ValueError(
        "Missing API credentials. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
        "GEMINI_API_KEY in .env."
    )


def resolve_gemini_model(requested_model: str, gemini_key: str | None) -> str:
    """Normalize and validate Gemini model names for CrewAI native provider."""
    model = requested_model.strip()
    if model.startswith("models/"):
        model = model.split("models/", 1)[1]
    if model.startswith("gemini/"):
        model = model.split("gemini/", 1)[1]

    if not gemini_key:
        return model

    try:
        from google import genai

        client = genai.Client(api_key=gemini_key)
        available = {
            m.name.split("models/", 1)[1]
            for m in client.models.list()
            if getattr(m, "supported_actions", None)
            and "generateContent" in m.supported_actions
            and m.name.startswith("models/")
        }
    except Exception:
        return model

    if model in available:
        return model

    for candidate in ("gemini-2.0-flash", "gemini-flash-latest"):
        if candidate in available:
            return candidate

    return model


def build_rag_tools(past_cases_dir: Path) -> list[Any]:
    """Create RAG tools with OpenAI-backed RagTool when available."""
    tools: list[Any] = []
    openai_key = os.getenv("OPENAI_API_KEY")

    # CrewAI RagTool/DirectorySearchTool currently require OPENAI_API_KEY
    # in this dependency version.
    if openai_key:
        tools.append(DirectorySearchTool(directory=str(past_cases_dir)))
        rag_tool = RagTool()
        # Example ingestion requested: index all known prior case files.
        rag_tool.add(data_type="directory", path=str(past_cases_dir))
        tools.insert(0, rag_tool)

    return tools


def build_agents(document_path: str, llm: LLM, rag_tools: list[Any]) -> dict[str, Agent]:
    """Create all agents for legal discovery processing."""
    parser_tools = [FileReadTool(file_path=document_path)]
    openai_key = os.getenv("OPENAI_API_KEY")

    # PDF/Directory semantic tools can require OpenAI embeddings in some builds.
    if openai_key:
        parser_tools.append(PDFSearchTool(pdf=str(document_path)))
        parser_tools.append(
            DirectorySearchTool(directory=str(Path(document_path).parent))
        )

    document_parser = Agent(
        role="DocumentParser",
        goal=(
            "Extract clean legal text, entities, and metadata from uploaded PDFs "
            "for solo attorneys practicing in Washington, DC."
        ),
        backstory=(
            "You are a litigation support analyst focused on DC solo firms. You are "
            "excellent at OCR-aware extraction and can normalize noisy exhibit scans, "
            "emails, and contract attachments into machine-readable facts."
        ),
        llm=llm,
        tools=parser_tools,
        allow_delegation=False,
        verbose=True,
    )

    risk_scanner = Agent(
        role="RiskScanner",
        goal=(
            "Detect legal and compliance risk, including PII exposure, fraud signals, "
            "attorney-client privilege concerns, and DC/federal regulatory issues."
        ),
        backstory=(
            "You are a forensic legal risk reviewer for solo attorneys handling civil "
            "and discovery-heavy matters. You compare current facts to prior similar "
            "matters to reduce blind spots and prioritize remediation."
        ),
        llm=llm,
        tools=rag_tools,
        allow_delegation=False,
        verbose=True,
    )

    case_brief_writer = Agent(
        role="CaseBriefWriter",
        goal=(
            "Produce a structured case brief with timeline, issues, risks, gaps, "
            "and concrete next actions grounded in past similar cases."
        ),
        backstory=(
            "You are a senior legal strategist supporting overwhelmed DC solo "
            "practitioners. You summarize complex records into practical, "
            "court-ready action plans."
        ),
        llm=llm,
        tools=rag_tools,
        allow_delegation=False,
        verbose=True,
    )

    return {
        "document_parser": document_parser,
        "risk_scanner": risk_scanner,
        "case_brief_writer": case_brief_writer,
    }


def build_tasks(agents: dict[str, Agent], document_path: str) -> list[Task]:
    """Create short-name sequential tasks with structured output enabled."""
    parse_doc = Task(
        name="parse_doc",
        description=(
            "Extract text and evidence details from {document_path}. "
            "Use OCR-capable PDF tools when needed. Return entities for person, date, "
            "org, and location; include source metadata and potential extraction gaps."
        ),
        expected_output=(
            "JSON with keys: metadata, extracted_text_summary, entities, exhibits, "
            "quality_warnings."
        ),
        agent=agents["document_parser"],
        output_json=True,
    )

    scan_risks = Task(
        name="scan_risks",
        description=(
            "Review parsed document output and flag: PII requiring redaction, fraud "
            "indicators, privilege concerns, FAR/DC Code issues, and litigation "
            "exposure severity. Pull analogous past-case insights from RAG."
        ),
        expected_output=(
            "JSON with keys: pii_findings, fraud_indicators, privilege_flags, "
            "regulatory_issues, risk_scores, similar_cases."
        ),
        agent=agents["risk_scanner"],
        context=[parse_doc],
        output_json=True,
    )

    write_brief = Task(
        name="write_brief",
        description=(
            "Generate a final structured case brief for the attorney. Include "
            "timeline, parties, key issues, critical risks, missing elements, "
            "and prioritized next_actions with rationale grounded in parsed facts "
            "and similar prior matters."
        ),
        expected_output=(
            "JSON with keys: case_summary, timeline, parties, key_issues, "
            "critical_risks, missing_elements, next_actions."
        ),
        agent=agents["case_brief_writer"],
        context=[parse_doc, scan_risks],
        output_json=True,
    )

    for task in (parse_doc, scan_risks, write_brief):
        task.description = task.description.format(document_path=document_path)

    return [parse_doc, scan_risks, write_brief]


def run_crew(document_path: str, document_text: str | None = None) -> Any:
    """Run the Legal Discovery AI crew against a single legal document."""
    ensure_supported_python()
    load_dotenv(PROJECT_ROOT / ".env")
    llm = build_llm()
    rag_tools = build_rag_tools(PAST_CASES_DIR)
    agents = build_agents(document_path=document_path, llm=llm, rag_tools=rag_tools)
    tasks = build_tasks(agents=agents, document_path=document_path)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True,
    )

    inputs = {
        "document_path": document_path,
        "document_text": document_text
        or "No direct document_text supplied. Agents should read from document_path.",
    }
    return crew.kickoff(inputs=inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Legal Discovery AI Crew.")
    parser.add_argument(
        "--document-path",
        default=str(PROJECT_ROOT / "data" / "sample.pdf"),
        help="Path to a legal PDF for processing.",
    )
    parser.add_argument(
        "--document-text",
        default=None,
        help="Optional text if you want to pass plain text content directly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_crew(document_path=args.document_path, document_text=args.document_text)
    try:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except TypeError:
        print(result)
