from math import exp, factorial
import pandas as pd
import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * lam**k / factorial(k)


def poisson_prediction(home_xg: float, away_xg: float, max_goals: int = 8) -> dict:
    home = np.array([poisson_pmf(i, home_xg) for i in range(max_goals + 1)])
    away = np.array([poisson_pmf(i, away_xg) for i in range(max_goals + 1)])
    matrix = np.outer(home, away)
    return {
        "home_win_probability": float(np.tril(matrix, -1).sum()),
        "draw_probability": float(np.trace(matrix)),
        "away_win_probability": float(np.triu(matrix, 1).sum()),
    }


def estimate_xg(shots: int, sot: int, goals: int) -> float:
    if shots <= 0:
        return 0.0
    return round(0.02 * shots + 0.08 * sot + 0.05 * goals, 3)


def match_analysis(row: pd.Series) -> dict:
    return {
        "match_id": row["match_id"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "score": f'{int(row["home_goals"])}-{int(row["away_goals"])}',
        "home_xg_proxy": estimate_xg(int(row["home_shots"]), int(row["home_sot"]), int(row["home_goals"])),
        "away_xg_proxy": estimate_xg(int(row["away_shots"]), int(row["away_sot"]), int(row["away_goals"])),
        "shot_difference": int(row["home_shots"] - row["away_shots"]),
        "sot_difference": int(row["home_sot"] - row["away_sot"]),
        "possession_difference": round(float(row["home_possession"] - row["away_possession"]), 2),
        "ppda_difference": round(float(row["home_ppda"] - row["away_ppda"]), 2),
        "final_third_entry_difference": int(row["home_final_third_entries"] - row["away_final_third_entries"]),
        "set_piece_difference": int(row["home_set_pieces"] - row["away_set_pieces"]),
    }


def team_elo(df: pd.DataFrame, k: float = 20.0) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, r in df.sort_values("date").iterrows():
        h, a = r["home_team"], r["away_team"]
        ratings.setdefault(h, 1500.0)
        ratings.setdefault(a, 1500.0)
        expected_h = 1 / (1 + 10 ** ((ratings[a] - ratings[h]) / 400))
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        actual_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        ratings[h] += k * (actual_h - expected_h)
        ratings[a] += k * ((1 - actual_h) - (1 - expected_h))
    return dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_binary(y_true, y_prob, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
