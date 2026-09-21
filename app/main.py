from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .data import load_matches, matches_frame
from .analytics import match_analysis, poisson_prediction, team_elo
from .models import PredictionRequest, Prediction

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Football Analysis System 2026", version="0.1.0")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "football-analysis-system"}


@app.get("/api/matches")
def matches():
    return [m.model_dump() for m in load_matches()]


@app.get("/api/matches/{match_id}/analysis")
def analysis(match_id: str):
    df = matches_frame()
    rows = df[df["match_id"] == match_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Match not found")
    return match_analysis(rows.iloc[0])


@app.post("/api/predict", response_model=Prediction)
def predict(req: PredictionRequest):
    df = matches_frame()
    home = df[df["home_team"] == req.home_team]
    away = df[df["away_team"] == req.away_team]
    if home.empty or away.empty:
        raise HTTPException(status_code=404, detail="Team not found in demo data")

    home_xg = max(0.2, float(home["home_goals"].mean()) * 0.65 + float(away["away_goals"].mean()) * 0.35)
    away_xg = max(0.2, float(away["away_goals"].mean()) * 0.65 + float(home["home_goals"].mean()) * 0.35)
    probs = poisson_prediction(home_xg, away_xg)

    return Prediction(
        home_team=req.home_team,
        away_team=req.away_team,
        home_xg=round(home_xg, 3),
        away_xg=round(away_xg, 3),
        model="poisson",
        **{k: round(v, 6) for k, v in probs.items()},
    )


@app.get("/api/teams/elo")
def elo():
    return team_elo(matches_frame())
