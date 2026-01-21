# Models Performance Report (Jan 2026)

Trained using High-Precision "Safe Win" Target (Hazard < 0.10 + Profit).

| Instrument | Precision | Recall | Imbalance | Note |
|------------|-----------|--------|-----------|------|
| **EUR_USD**| 31.6%     | 99.3%  | 75x       | Excellent |
| **GBP_USD**| 30.9%     | 97.5%  | 52x       | Excellent |
| **USD_JPY**| 28.0%     | 99.4%  | 25x       | High Frequency Wins |
| **USD_CHF**| 32.8%     | 86.0%  | 235x      | High Precision, Rare |
| **USD_CAD**| 23.2%     | 83.7%  | 187x      | Solid |
| **AUD_USD**| 19.3%     | 70.7%  | 310x      | Harder Regime |
| **NZD_USD**| 17.6%     | 88.2%  | 302x      | Low Frequency |

## Deployment
Models are located in this directory.
Loaded by `scripts/trading/regime_agent.py` automatically based on instrument name.
