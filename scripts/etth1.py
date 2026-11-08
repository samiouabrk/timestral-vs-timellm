"""Timestral zero-shot on ETTh1 under the Time-LLM long-term protocol (input 512, H in {96,192,336,720},
standard 12/4/4-month split, train-set standardisation, MSE/MAE over all 7 channels and every test window)."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster

torch.set_num_threads(8)
OUT = Path(__file__).resolve().parent.parent
L, HS, HMAX = 512, [96, 192, 336, 720], 720
df = pd.read_csv(f"{OUT}/data/ETTh1.csv")
X = df.iloc[:, 1:].to_numpy(float).T                         # (7, T)
tr_end, te_start, te_end = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
mu, sd = X[:, :tr_end].mean(1, keepdims=True), X[:, :tr_end].std(1, keepdims=True)
Z = (X - mu) / sd
n_win = te_end - te_start - min(HS) + 1                      # 2785 windows (H=96 set)
starts = te_start + np.arange(n_win)

TIMELLM = {96: (0.362, 0.392), 192: (0.398, 0.418), 336: (0.430, 0.427), 720: (0.442, 0.457)}
PATCHTST = {96: (0.370, 0.399), 192: (0.413, 0.421), 336: (0.422, 0.436), 720: (0.447, 0.466)}

def score(predZ):                                            # predZ: (n_win, 7, HMAX) scaled
    res = {}
    for H in HS:
        k = te_end - te_start - H + 1                        # windows valid for this horizon
        tru = np.stack([Z[:, s:s + H] for s in starts[:k]])
        p = predZ[:k, :, :H]
        res[H] = (float(((p - tru) ** 2).mean()), float(np.abs(p - tru).mean()))
    res["avg"] = tuple(float(np.mean([res[H][i] for H in HS])) for i in (0, 1))
    return res

if __name__ == "__main__":
    sizes = sys.argv[1:] or ["5M", "28M", "209M", "123M"]
    results = {}
    # baselines (scaled space)
    results["Naive (repeat last)"] = score(np.stack([np.repeat(Z[:, s - 1:s], HMAX, 1) for s in starts]))
    results["Seasonal naive (24h)"] = score(np.stack([np.tile(Z[:, s - 24:s], (1, HMAX // 24)) for s in starts]))
    for size in sizes:
        fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")
        if size == sizes[0]:                                  # prefix check: 720-step forecast vs 96-step forecast
            c = np.stack([X[:, s - L:s] for s in starts[:2]])
            a = np.asarray(fc.forecast(c, prediction_length=HMAX).median)[..., :96]
            b = np.asarray(fc.forecast(c, prediction_length=96).median)
            print(f"prefix check max abs diff: {np.abs(a - b).max():.2e}", flush=True)
        t, preds = time.time(), []
        for j in range(0, n_win, 16):
            c = np.stack([X[:, s - L:s] for s in starts[j:j + 16]])
            preds.append(np.asarray(fc.forecast(c, prediction_length=HMAX).median))
        P = np.concatenate(preds)
        np.save(f"{OUT}/etth1_timestral_{size}_median.npy", P)
        results[f"Timestral-{size} (zero-shot)"] = score((P - mu[None]) / sd[None])
        print(f"Timestral-{size}: {time.time() - t:.0f}s  avg MSE/MAE {results[f'Timestral-{size} (zero-shot)']['avg']}", flush=True)
        json.dump(results, open(f"{OUT}/results_etth1.json", "w"), indent=1)
        del fc
    results["Time-LLM (paper, fine-tuned Llama-7B)"] = {**TIMELLM, "avg": (0.408, 0.423)}
    results["PatchTST (as reported in Time-LLM paper)"] = {**PATCHTST, "avg": (0.413, 0.430)}
    json.dump(results, open(f"{OUT}/results_etth1.json", "w"), indent=1)
    rows = {m: {f"{H} MSE": r[H][0] for H in HS + ["avg"]} | {f"{H} MAE": r[H][1] for H in HS + ["avg"]} for m, r in results.items()}
    print(pd.DataFrame(rows).T.round(3).to_string())
