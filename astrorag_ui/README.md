# AstroRAG UI

Streamlit web interface for AstroRAG. **Read-only** display of pipeline results
generated on the SDSU JupyterHub server.

## Live demo

Deployed at: https://astrorag.streamlit.app _(update after deployment)_

## Contents

- **Papers page** — browse all processed papers with equations and numerical results
- **Chat page** — ask follow-up questions about any paper (calls Groq LLM)
- **Benchmark page** — view 20-query evaluation and 6-variant ablation results

## Local development

```bash
pip install -r requirements.txt

# set your Groq API key
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your key

streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select repo and set main file: `streamlit_app.py`
6. Under "Advanced settings" → "Secrets", add:
```
   GROQ_API_KEY = "gsk_..."
```
7. Click "Deploy"

## Data

The `data/` directory contains:

- `context_library.json` — all processed papers (main data source)
- `eval_full.json` — 20-query benchmark results
- `ablation/ablation_*.json` — 6-variant ablation traces

To update the demo with new results, copy fresh files from the SDSU server
and push to GitHub. Streamlit Cloud auto-redeploys.
