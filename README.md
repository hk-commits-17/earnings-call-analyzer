# 📊 Earnings Call Analyzer

AI-powered first-pass analysis of quarterly earnings calls with quarter-over-quarter sentiment comparison.

Built with FinBERT for quantitative sentiment scoring and Claude for interpretation.

## What it does

Upload or fetch two quarterly earnings call transcripts and get:

1. **Key Themes** — top 5-7 topics with descriptions
2. **Sentiment Heatmap** — FinBERT scores per topic per quarter, visualised as a grouped bar chart
3. **Sentiment Narrative** — LLM interpretation of the biggest shifts between quarters
4. **Risk Flags** — hedging language, evasive answers, guidance red flags with severity ratings
5. **Q&A Summary** — each analyst exchange condensed with quality tags (Direct Answer / Partial Answer / Deflection)
6. **Questions I'd Ask** — AI-generated follow-up questions based on gaps and flags

## Setup

### 1. Clone and install

```bash
cd earnings-analyzer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add:
- `ANTHROPIC_API_KEY` — get one at https://console.anthropic.com
- `FMP_API_KEY` — get a free key at https://site.financialmodelingprep.com/developer/docs

### 3. Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

### Option A: Ticker lookup
Enter a US ticker symbol (e.g., AAPL, MSFT, NVDA). The app fetches the two most recent quarterly transcripts via the Financial Modeling Prep API.

### Option B: Upload files
Upload two transcript files (PDF or TXT) — one for the current quarter, one for the prior quarter.

Then click **Run Analysis** and wait for the pipeline to complete (typically 1-2 minutes).

## Architecture

```
app.py                          # Streamlit UI
analyzer/
  transcript_fetcher.py         # API + upload handling
  parser.py                     # Transcript → structured sections
  sentiment.py                  # FinBERT quantitative scoring
  llm_analysis.py               # Claude API (themes, risks, Q&A, interpretation)
  comparison.py                 # Quarter-over-quarter pipeline
  report.py                     # PDF export
config.py                       # API keys, model settings
```

### Analysis pipeline

1. **Fetch/parse** — transcript text is loaded and split into prepared remarks vs Q&A
2. **Topic extraction** — Claude extracts text passages for fixed categories (Revenue, Margins, Guidance, Competitive Positioning, Capital Allocation, Risk Factors) plus emergent topics
3. **Sentiment scoring** — FinBERT scores each topic passage (-1 to +1)
4. **Theme extraction** — Claude identifies the 5-7 most important themes
5. **Risk flagging** — Claude flags evasive language, guidance signals, red flag phrases, and structural tells
6. **Q&A summarisation** — Claude condenses each analyst exchange with a quality tag
7. **Comparison** — sentiment scores are diffed across quarters, Claude interprets the shifts
8. **Question generation** — Claude generates pointed follow-up questions based on gaps and flags

## Risk flag taxonomy

The tool flags four categories of concern:

- **Evasive language**: non-answers, qualitative shifts, "monitoring" language without specifics
- **Guidance signals**: widened ranges, dropped metrics, redefined non-GAAP measures
- **Red flag phrases**: recurring "one-time" charges, macro blame, "investing for the long term" during margin compression
- **Structural tells**: wrong exec answering, short Q&A sessions, cut-off follow-ups

## Limitations

- Transcript parsing is best-effort — formats vary across providers
- FinBERT has a 512-token context window; long passages are truncated
- Sentiment scores are relative, not absolute — useful for comparison, not as standalone metrics
- Claude API costs apply (~$0.10-0.30 per full analysis run)
- US-listed companies only via API; other markets require manual upload

## Tech stack

- Python 3.11+
- Streamlit (UI)
- Anthropic Claude API (analysis)
- FinBERT / HuggingFace Transformers (sentiment)
- Financial Modeling Prep API (transcripts)
- ReportLab (PDF generation)
- Plotly (charts)

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `.env` is not committed — it's in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **New app** → select your repo, branch `main`, file `app.py`

4. In your app's **Settings → Secrets**, add:

```toml
ANTHROPIC_API_KEY = "your_key_here"
FMP_API_KEY = "your_key_here"
```

5. Deploy — Streamlit Cloud will install dependencies and start the app

**Note on memory:** FinBERT requires ~400MB of model weights. If Streamlit Cloud's 1GB memory limit causes issues, the app automatically falls back to a keyword-based sentiment scorer. The keyword fallback is less accurate but keeps the app running.
