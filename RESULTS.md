# Timestral (zero-shot) vs Time-LLM (fine-tuned) on daily BTC — 2026-11-08

**Setup.** Test split of the [Time-LLM study](https://github.com/samiouabrk/Time-LLM-Cryptex) (last 20% of `candlesticks-D.csv`): 325 windows, targets 2023-06-07 → 2024-05-09, 14-day horizon, close price.
Time-LLM scored from its saved forecasts (`timellm_forecasts/`), no re-run.
Timestral run zero-shot on CPU (`scripts/benchmark.py`). Metrics: MAE ($), MASE (vs in-sample 1-step naive on the 96-day context), CRPS (quantile approximation; = MAE for point forecasts), MDA.

| Model | MAE | MASE | CRPS | MDA |
|---|---:|---:|---:|---:|
| Naive (last value) | 2,078 | 3.765 | 2,078 (point) | n/a |
| Random walk + empirical quantiles, full history | 2,064 | — | 1,493 | — |
| ARIMA(1,1,1), 96-day context | 2,080 | 3.793 | 2,080 (point) | 0.500 |
| Time-LLM Llama 3.2 3B LoRA r16 (best) | 2,629 | 4.821 | 2,629 (point) | 0.452 |
| Time-LLM Llama 3.2 3B frozen | 2,856 | 5.186 | 2,856 (point) | 0.452 |
| Time-LLM Qwen2.5 3B LoRA r8 | 3,568 | 6.571 | 3,568 (point) | 0.481 |
| Timestral-5M, full history | 2,081 | 3.759 | 1,516 | 0.535 |
| Timestral-28M, full history | 2,081 | 3.774 | 1,517 | 0.491 |
| Timestral-123M, full history | 2,069 | 3.740 | 1,500 | 0.547 |
| **Timestral-209M, full history** | **2,063** | **3.734** | **1,494** | 0.548 |
| Timestral-209M, full history + OHLV past covariates | 2,065 | 3.738 | 1,492 | 0.539 |
| Timestral-209M, 96-day context | 2,131 | 3.859 | 1,563 | 0.491 |

## Findings

1. **No Time-LLM configuration beats a naive forecast.** The best (Llama r16) is 27% worse than repeating the last price.
2. **Timestral zero-shot matches the random walk** on point error (−0.7% vs naive) and on CRPS (1,494 vs 1,493 for a random walk with empirical quantiles). Daily BTC is close to unpredictable at this horizon, and Timestral behaves like a calibrated random walk instead of inventing structure.
3. Longer context helps Timestral (all sizes); covariates (open/high/low/volume) add nothing here, as expected (largely redundant with close).
4. MDA differences (0.45–0.55) are not meaningful: 325 heavily overlapping windows.

## Caveats

- The Time-LLM training logs report MAE averaged over all 5 OHLCV channels (incl. volume), which is lower than the close-only MAE used here (Llama r16: 2,325 vs 2,629). On close price, LoRA r16 improves Llama 3.2 3B by 7.9% over its frozen backbone (2,856 → 2,629).
- Saved Time-LLM forecasts were produced with fp16 autocast on GPU.

---

## ETTh1, long-term forecasting (Time-LLM protocol)

Input 512, horizons 96/192/336/720, train/val/test 12/4/4 months, z-scored with training statistics.
MSE/MAE averaged over all 7 channels and all 2,785 test windows. Script: `scripts/etth1.py`.
Timestral: one 720-step forecast per window, scored on each prefix (720-step vs 96-step prefix differ by at most 9e-6, so this is equivalent).

| Model | 96 MSE | 192 | 336 | 720 | **avg MSE** | 96 MAE | 192 | 336 | 720 | **avg MAE** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive (repeat last) | 1.294 | 1.325 | 1.330 | 1.335 | 1.321 | 0.713 | 0.733 | 0.746 | 0.755 | 0.737 |
| Seasonal naive (24h) | 0.512 | 0.581 | 0.650 | 0.655 | 0.600 | 0.433 | 0.469 | 0.501 | 0.514 | 0.479 |
| Timestral-5M, zero-shot | 0.368 | 0.409 | 0.428 | 0.429 | 0.409 | 0.376 | 0.403 | 0.417 | 0.441 | 0.409 |
| Timestral-28M, zero-shot | 0.354 | 0.401 | 0.428 | 0.451 | 0.409 | 0.370 | 0.401 | 0.418 | 0.445 | 0.408 |
| Timestral-123M, zero-shot | 0.354 | 0.399 | 0.421 | 0.433 | 0.402 | 0.368 | 0.398 | 0.415 | 0.437 | 0.405 |
| **Timestral-209M, zero-shot** | 0.354 | 0.399 | 0.419 | 0.426 | **0.400** | 0.367 | 0.397 | 0.412 | 0.433 | **0.402** |
| Time-LLM, Llama-7B, trained on ETTh1 (paper) | 0.362 | 0.398 | 0.430 | 0.442 | 0.408 | 0.392 | 0.418 | 0.427 | 0.457 | 0.423 |
| PatchTST, trained on ETTh1 (as reported by Time-LLM) | 0.370 | 0.413 | 0.422 | 0.447 | 0.413 | 0.399 | 0.421 | 0.436 | 0.466 | 0.430 |

CPU runtime for the whole test set: 5M 106 s · 28M 329 s · 123M 1,191 s · 209M 2,195 s.

**Findings**
- Timestral-209M zero-shot beats Time-LLM's ETTh1-trained result: avg MSE 0.400 vs 0.408 (−2%), avg MAE 0.402 vs 0.423 (−5%). It wins on MAE at every horizon and on MSE at 3 of 4 horizons (192: 0.399 vs 0.398).
- Even Timestral-5M (zero-shot, ~1,400× fewer parameters than Llama-7B) matches Time-LLM on MSE (0.409 vs 0.408) and beats it on MAE (0.409 vs 0.423).
- Time-LLM's advantage over PatchTST (0.408 vs 0.413) is smaller than the gap between Time-LLM and a zero-shot 209M native model.

**Caveats**
- Time-LLM and PatchTST numbers are copied from the Time-LLM paper, not re-run here.
- The Timestral report says post-training used real data but does not list the datasets, so overlap with ETTh1 cannot be ruled out. ETT is part of the GIFT-Eval test suite, which foundation-model authors usually hold out, but this is not confirmed.
- One dataset; ETTh1 is small and widely used as a benchmark.
