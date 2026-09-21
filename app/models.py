from pydantic import BaseModel, Field
from typing import Literal


class Match(BaseModel):
    match_id: str
    date: str
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    home_shots: int = Field(ge=0)
    away_shots: int = Field(ge=0)
    home_sot: int = Field(ge=0)
    away_sot: int = Field(ge=0)
    home_possession: float = Field(ge=0, le=100)
    away_possession: float = Field(ge=0, le=100)
    home_ppda: float = Field(gt=0)
    away_ppda: float = Field(gt=0)
    home_final_third_entries: int = Field(ge=0)
    away_final_third_entries: int = Field(ge=0)
    home_set_pieces: int = Field(ge=0)
    away_set_pieces: int = Field(ge=0)


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class Prediction(BaseModel):
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    model: Literal["poisson"]
