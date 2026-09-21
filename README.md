# Football Analysis System 2026

A modular football analytics MVP with a FastAPI backend, statistical models, demo data, tests, and Docker support.

## Features
- Match, team, and player-ready data models
- Shot and xG proxy analysis
- Possession, PPDA, final-third entries, and set pieces
- Match-flow summaries
- W/D/L prediction with a Poisson baseline
- Elo ratings
- Brier score and log-loss utilities
- Replaceable data-provider layer
- FastAPI API and simple dashboard entry page
- PostgreSQL-ready Docker Compose
- GitHub Actions CI

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API
- `GET /health`
- `GET /api/matches`
- `GET /api/matches/{match_id}/analysis`
- `POST /api/predict`
- `GET /api/teams/elo`

The demo dataset is synthetic and intentionally small. Replace it with licensed/authorized data through the provider layer before production use.
