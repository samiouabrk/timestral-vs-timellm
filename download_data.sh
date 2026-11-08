#!/usr/bin/env bash
# Download the two public datasets used by the benchmark into ./data
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
# Daily BTC-USDT OHLCV (CRYPTEX BTC-D subset), as used in the Time-LLM study
curl -fsSL -o data/candlesticks-D.csv \
  https://raw.githubusercontent.com/abenhnini/Time-LLM-Cryptex/main/dataset/cryptex/candlesticks-D.csv
# ETTh1 (Zhou et al., 2021)
curl -fsSL -o data/ETTh1.csv \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
echo "Downloaded data/candlesticks-D.csv and data/ETTh1.csv"
