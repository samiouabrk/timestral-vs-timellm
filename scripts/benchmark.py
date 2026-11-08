"""Timestral (zero-shot) vs saved Time-LLM forecasts vs baselines, on the BTC test windows."""
import time, warnings, json
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from timestral import TimestralForecaster
from timellm_scores import raw, ends, truth, close, H, SEQ, load_preds, ROOT

warnings.filterwarnings("ignore")
QL = None

def metrics(point, q=None, levels=None):
    err = np.abs(point - truth)
    # MASE: scale by in-sample one-step naive MAE on the 96-day context of each window
    scale = np.array([np.abs(np.diff(close[e - SEQ + 1:e + 1])).mean() for e in ends])
    mase = (err.mean(1) / scale).mean()
    if q is None:
        crps = err.mean()                                   # CRPS of a point forecast = MAE
    else:
        lv = np.array(levels)[None, None, :]
        e = truth[..., None] - q
        pin = np.maximum(lv * e, (lv - 1) * e)
        crps = 2 * pin.mean()                               # quantile approximation of CRPS
    last = close[ends][:, None]
    mda = (np.sign(point - last) == np.sign(truth - last)).mean()
    return dict(MAE=err.mean(), MASE=mase, CRPS=crps, MDA=mda)

rows = {}
# Baselines
naive = np.repeat(close[ends][:, None], H, 1)
rows["Naive (last value)"] = metrics(naive)
t = time.time()
ar = np.stack([ARIMA(close[e - SEQ + 1:e + 1], order=(1, 1, 1)).fit().forecast(H) for e in ends])
rows["ARIMA(1,1,1), 96-day context"] = metrics(ar)
print(f"ARIMA done in {time.time()-t:.0f}s", flush=True)

# Saved Time-LLM forecasts (no re-run)
for m in ["llama32_3b_r16", "llama32_3b_frozen", "llama32_3b_huber", "gemma_1b_huber", "gemma_1b_baseline", "qwen25_3b_r8"]:
    rows[f"Time-LLM {m}"] = metrics(load_preds(m))

# Timestral zero-shot
feat = raw[["open", "high", "low", "volume"]].to_numpy(float)
for size in ["5M", "28M", "123M", "209M"]:
    fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")
    for ctx_name, ctx_len in [("96-day", SEQ), ("full history", None)]:
        L = SEQ if ctx_len else ends[-1] + 1
        ctx = np.full((len(ends), 1, L), np.nan)
        cov = np.full((len(ends), 4, L), np.nan)
        for i, e in enumerate(ends):
            s = 0 if ctx_len is None else e - SEQ + 1
            seg = close[s:e + 1]; ctx[i, 0, -len(seg):] = seg
            cov[i, :, -len(seg):] = feat[s:e + 1].T
        for use_cov in ([False, True] if size in ("28M", "209M") and ctx_len is None else [False]):
            t = time.time()
            B = 16 if (use_cov or ctx_len is None) else 128
            qs = []
            for j in range(0, len(ends), B):
                out = fc.forecast(ctx[j:j+B], prediction_length=H,
                                  past_covariates=cov[j:j+B] if use_cov else None)
                qq = out.quantiles
                qs.append((qq.numpy() if hasattr(qq, "numpy") else np.asarray(qq))[:, 0])
            q = np.concatenate(qs)
            med = q[..., list(out.quantile_levels).index(0.5)]
            name = f"Timestral-{size}, {ctx_name}" + (" + OHLV covariates" if use_cov else "")
            rows[name] = metrics(med, q, out.quantile_levels)
            print(f"{name}: {time.time()-t:.1f}s", flush=True)
            pd.DataFrame(rows).T.to_csv(ROOT / "results_btc.csv")
    del fc; import gc; gc.collect()

df = pd.DataFrame(rows).T
df.to_csv(ROOT / "results_btc.csv")
pd.set_option("display.width", 200)
print(df.round({"MAE": 0, "MASE": 3, "CRPS": 0, "MDA": 3}).to_string())
