from app.analytics import poisson_prediction, estimate_xg, team_elo
from app.data import matches_frame


def test_poisson_probabilities_sum_close_to_one():
    p = poisson_prediction(1.5, 1.0)
    assert abs(sum(p.values()) - 1.0) < 0.01


def test_xg_proxy_positive():
    assert estimate_xg(10, 4, 1) > 0


def test_elo_has_all_teams():
    ratings = team_elo(matches_frame())
    assert set(ratings) == {"North FC", "United FC", "City FC"}
