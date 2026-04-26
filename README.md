# Legal Discovery AI (CrewAI)

Production-oriented multi-agent legal discovery pipeline for DC solo attorneys.

## What it does

- Parses uploaded legal PDFs (contracts, emails, discovery documents).
- Extracts metadata + entities (person, date, org, location), including OCR-friendly parsing via CrewAI PDF tools.
- Flags legal/compliance risks:
  - PII needing redaction
  - fraud indicators
  - attorney-client privilege concerns
  - FAR / DC Code risk signals
- Generates a structured case brief with:
  - timeline
  - parties
  - key issues
  - critical risks
  - missing elements
  - actionable next steps
- Uses simple RAG over `data/past_cases/` so agents can ground recommendations in similar prior matters.

## Project structure

```text
legal_discovery_ai/
  .env
  .env.example
  data/past_cases/
  data/uploads/
  src/legal_discovery_ai/app.py
  src/legal_discovery_ai/crew.py
  pyproject.toml
  requirements.txt
```

## Setup

1) Install and use Python 3.11 or 3.12 (Python 3.14 is not supported yet by some CrewAI dependencies on Windows).

2) Create a virtual environment and install dependencies:

```powershell
cd "C:\Users\12404\Downloads\Mercy AI\legal_discovery_ai"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If you plan to use Gemini, the project now installs CrewAI with the Google GenAI provider via:
`crewai[tools,google-genai]`.

3) Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

4) Add one API key (Anthropic preferred, Gemini supported):

```env
ANTHROPIC_API_KEY=your_key_here
# OR
# OPENAI_API_KEY=your_key_here
# OR
# GEMINI_API_KEY=your_key_here
# Optional override if your account has a different available model:
# GEMINI_MODEL=gemini-2.0-flash
```

## Run

CLI:

```powershell
python -m legal_discovery_ai.crew --document-path "data\sample.pdf"
```

Optional:

```powershell
python -m legal_discovery_ai.crew --document-path "data\sample.pdf" --document-text "Optional supplemental context"
```

Streamlit UI:

```powershell
streamlit run "src/legal_discovery_ai/app.py"
```

UI includes:
- Attorney-facing formatted report sections (not just raw JSON)
- Per-agent status panel with estimated progress through parser, scanner, and brief writer
- One-click export selector for Markdown, PDF, or JSON

If you previously created a `.venv` with Python 3.14, delete it and recreate with Python 3.12.

## Add test PDFs from CourtListener (free source)

1) Open [CourtListener Opinion Search](https://www.courtlistener.com/).
2) Search for public opinions relevant to your practice area.
3) Open a result and download the PDF.
4) Save files into:

```text
data/past_cases/
```

You can also place your active case file in `data/` (for example `data/sample.pdf`) and run the crew against it.

## Sample test command

```powershell
python -m legal_discovery_ai.crew --document-path "data\past_cases\example_case.pdf"
```

## Notes

- LLM preference order is Anthropic Claude 3.5 Sonnet, then OpenAI GPT-4o, then Gemini (`GEMINI_MODEL`, default `gemini-2.0-flash`).
- RAG directory ingestion happens in code using:
  - `rag_tool.add(data_type="directory", path="data/past_cases/")`
- `RagTool` (vector RAG) is enabled when `OPENAI_API_KEY` is present.
- In this CrewAI tools version, `RagTool`, `DirectorySearchTool`, and `PDFSearchTool` may require `OPENAI_API_KEY`; Gemini-only mode falls back to `FileReadTool`-based processing so the workflow still runs.
- Crew execution is configured as `Process.sequential`, `verbose=True`, and `memory=True`.
