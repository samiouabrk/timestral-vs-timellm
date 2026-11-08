"""Score saved Time-LLM forecasts (close price) on the test windows of the BTC study (no re-run)."""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "timellm_forecasts"
SEQ, H = 96, 14

raw = pd.read_csv(ROOT / "data" / "candlesticks-D.csv")
raw["date"] = pd.to_datetime(raw["timestamp"], unit="s").dt.normalize()
N = len(raw)
num_test = int(N * 0.2)
first_end = N - num_test - 1                      # last context row of the first test window
ends = np.arange(first_end, N - H)                 # last context row of each test window
close = raw["close"].to_numpy()
truth = np.stack([close[e + 1:e + 1 + H] for e in ends])   # (windows, H)

def load_preds(model):
    df = pd.read_csv(RES / f"{model}_inference.csv")
    df = df.dropna(subset=["close_predicted_1"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize() if df["timestamp"].dtype == object \
        else pd.to_datetime(df["timestamp"], unit="s").dt.normalize()
    df = df.drop_duplicates("date", keep="last").set_index("date")
    cols = [f"close_predicted_{j}" for j in range(1, H + 1)]
    return np.stack([df.loc[raw["date"].iloc[e], cols].to_numpy(float) for e in ends])

if __name__ == "__main__":
    print(f"rows={N} test windows={len(ends)} first target {raw['date'].iloc[ends[0]+1].date()} "
          f"last target {raw['date'].iloc[ends[-1]+H].date()}")
    for f in sorted(RES.glob("*_inference.csv")):
        m = f.name.replace("_inference.csv", "")
        print(f"{m:20s} close-price MAE = {np.abs(load_preds(m) - truth).mean():8.0f}")
