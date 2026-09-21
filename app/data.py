from pathlib import Path
import pandas as pd
from .models import Match

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_matches.csv"


def load_matches() -> list[Match]:
    df = pd.read_csv(DATA_PATH)
    return [Match(**row.to_dict()) for _, row in df.iterrows()]


def matches_frame() -> pd.DataFrame:
    return pd.DataFrame([m.model_dump() for m in load_matches()])
