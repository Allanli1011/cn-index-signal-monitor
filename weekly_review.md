## Weekly Review — 2026-10-31

_Window: rolling Sharpe over last 30 trades; loss-streak window 10 trades; decay alert if realised < 50% of historical_

### 🚨 Alerts
- **IM_fade_short**: RECENT_LOSS_STREAK

### Performance summary
| Strategy | Trades | Hit % | Mean bps | Realised Sharpe | Rolling Sharpe | Historical | Cum PnL | Last10 PnL | Alerts |
|----------|--------|-------|----------|----------------|---------------|------------|---------|-----------|--------|
| IH_fade_short | 6 | 33.3% | -46.7 | -1.665 | -1.665 | 0.76 | -47,968 | -47,968 | — |
| IF_fade_short | 6 | 33.3% | -70.6 | -2.701 | -2.701 | 0.68 | -58,637 | -58,637 | — |
| IC_fade_short | 3 | 33.3% | -133.8 | -2.994 | — | 0.49 | -61,461 | -61,461 | — |
| IM_fade_short | 19 | 52.6% | -2.7 | -0.102 | -0.102 | 0.74 | -1,029 | -46,192 | RECENT_LOSS_STREAK |

_Charts: rolling_sharpe.png + cumulative_pnl.png in repo_