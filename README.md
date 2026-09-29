# 🤖 AI Data Analyst Agent

An LLM agent (tool use) that plans, calls safe analysis tools on any CSV, and explains results with a full, inspectable trace.
Combines **LLM agents, AutoML, explainability, anomaly detection and AI safety guardrails** in one deployed app.

**Live demo:** _add your Streamlit link here_

## What it does
Polished dark UI with gradient hero, KPI cards, interactive Altair charts, one-click suggested questions and a visible agent trace.

- **Agent chat:** ask in plain English ("Which features best predict cultivar? Plot alcohol by cultivar"). The agent chooses tools,
  reads results, self-corrects on errors and answers with real numbers. Bring-your-own Anthropic key (never stored).
- **Auto-analysis (no key needed):** profiling, top correlations, Isolation Forest outliers, and an AutoML comparison
  (baseline vs linear vs random forest, cross-validated) with permutation feature importance.
- **Tools:** `describe_data`, `aggregate`, `correlations`, `detect_outliers`, `plot`, `train_model`

## Guardrails (AI engineering)
- No `eval`/`exec`: typed tool arguments, validated column names, whitelisted filter operators
- Step limit, row cap, output truncation; tool errors are fed back so the model can recover
- Dataset content treated as untrusted (prompt-injection defence)
- Agent loop tested with a scripted fake LLM: deterministic, no network, no API key in CI

## Sample results (bundled wine dataset, 5-fold CV macro-F1)
Baseline 0.19 → Logistic Regression 0.98 (best), Random Forest 0.97. Top features: hue, proline, flavanoids.
Regression path checked on the diabetes dataset (best CV R² ≈ 0.48 with Ridge).
🚀 Live Demo:
https://muskan136-demand-forecast-ai-app-hrjbbb.streamlit.app/

An AI-powered demand forecasting application for analyzing historical demand data and generating future forecasts.

## Run
```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements-dev.txt
pytest -q
python -m streamlit run app.py
```
Docker: `docker build -t ai-agent . && docker run -p 8501:8501 ai-agent`

## Structure
`app.py` UI · `agent.py` tool-use loop · `tools.py` safe tools + schemas + AutoML · `tests/` · CI · Dockerfile
