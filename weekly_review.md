## Weekly Review — 2026-08-07

_Window: rolling Sharpe over last 30 trades; loss-streak window 10 trades; decay alert if realised < 50% of historical_

### 🚨 Alerts
- **IM_fade_short**: RECENT_LOSS_STREAK

### Performance summary
| Strategy | Trades | Hit % | Mean bps | Realised Sharpe | Rolling Sharpe | Historical | Cum PnL | Last10 PnL | Alerts |
|----------|--------|-------|----------|----------------|---------------|------------|---------|-----------|--------|
| IH_fade_short | 5 | 40.0% | -49.9 | -1.597 | -1.597 | 0.76 | -42,739 | -42,739 | — |
| IF_fade_short | 5 | 20.0% | -87.6 | -3.314 | -3.314 | 0.68 | -60,588 | -60,588 | — |
| IC_fade_short | 3 | 33.3% | -133.8 | -2.994 | — | 0.49 | -61,461 | -61,461 | — |
| IM_fade_short | 12 | 50.0% | -22.7 | -0.826 | -0.826 | 0.74 | -35,188 | -78,164 | RECENT_LOSS_STREAK |

_Charts: rolling_sharpe.png + cumulative_pnl.png in repo_