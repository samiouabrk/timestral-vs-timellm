# Timestral vs Time-LLM

Two ways to use foundation models for forecasting:

- **Reprogramming an LLM** ([Time-LLM](https://arxiv.org/abs/2310.01728)): a language model backbone, with patch embeddings mapped into its token space and optionally LoRA fine-tuning.
- **A native time-series foundation model** ([Timestral](https://arxiv.org/abs/2610.37585), Mistral AI): a small model pre-trained on time series, used zero-shot.

I had fine-tuned Time-LLM on daily Bitcoin in 14 configurations ([Time-LLM-Cryptex](https://github.com/samiouabrk/Time-LLM-Cryptex)). This repo benchmarks Timestral zero-shot against that work and against Time-LLM's published results, on CPU.

## Results

### ETTh1 (Time-LLM long-term protocol)

Input 512, horizons 96/192/336/720, standard 12/4/4-month split, z-scored with training statistics, MSE/MAE over all 7 variates and all test windows. Time-LLM and PatchTST numbers are taken from the Time-LLM paper.

| Model | avg MSE | avg MAE |
|---|---:|---:|
| Seasonal naive (24h) | 0.600 | 0.479 |
| Timestral-5M, zero-shot | 0.409 | 0.409 |
| Timestral-28M, zero-shot | 0.409 | 0.408 |
| Timestral-123M, zero-shot | 0.402 | 0.405 |
| **Timestral-209M, zero-shot** | **0.400** | **0.402** |
| Time-LLM, Llama-7B trained on ETTh1 (paper) | 0.408 | 0.423 |
| PatchTST, trained on ETTh1 (as reported in Time-LLM paper) | 0.413 | 0.430 |

Zero-shot, Timestral-209M beats the fine-tuned Llama-7B Time-LLM on average MSE and MAE, and the 5M model already matches it.

### Daily BTC, 14-day horizon (close price)

Test split of the Time-LLM study: 325 windows, targets June 2023 – May 2024. Time-LLM is scored from its saved forecasts.

| Model | MAE ($) | MASE | CRPS |
|---|---:|---:|---:|
| Naive (last value) | 2,078 | 3.765 | — |
| Random walk + empirical quantiles | 2,064 | — | 1,493 |
| ARIMA(1,1,1) | 2,080 | 3.793 | — |
| Time-LLM, Llama 3.2 3B LoRA r16 (best of 14) | 2,629 | 4.821 | — |
| **Timestral-209M, zero-shot, full history** | **2,063** | **3.734** | **1,494** |

No Time-LLM configuration beats a naive forecast. Timestral matches a calibrated random walk, which is the right answer for daily BTC at this horizon: it does not invent structure.

Per-horizon tables, all model sizes, covariate runs and caveats: [RESULTS.md](RESULTS.md).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh

cd scripts
python timellm_scores.py   # close-price MAE of the 14 saved Time-LLM forecasts
python benchmark.py        # BTC: baselines, Time-LLM, Timestral (all sizes, with and without covariates)
python etth1.py            # ETTh1: baselines and Timestral 5M/28M/209M/123M (about 1 h on CPU)
```

## Layout

```
scripts/timellm_scores.py   test windows and Time-LLM forecast loading (BTC)
scripts/benchmark.py        BTC benchmark -> results_btc.csv
scripts/etth1.py            ETTh1 benchmark -> results_etth1.json
timellm_forecasts/          saved Time-LLM forecasts from the 14 configurations
RESULTS.md                  full results and notes
```

## References

- Jin et al. *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.* ICLR 2024.
- Mistral AI. *Timestral* technical report, arXiv:2610.37585.
- Zhou et al. *Informer* (ETT dataset). AAAI 2021.
- CRYPTEX dataset: http://crypto.cs.iit.edu/datasets/download.html
