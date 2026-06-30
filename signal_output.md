## A-Share Index Futures Signal — 2026-06-30

**Capital basis:** 1,000,000 CNY

### Latest US session
| Index | Date | Close | Return | Age |
|-------|------|-------|--------|-----|
| SPX | 2026-06-29 | 7440.43 | +1.18% | 1d |
| NDX | 2026-06-29 | 25820.14 | +2.07% | 1d |

### Signals triggered: **3 / 4**

| Futures | Underlying | Action | Qty | Signal | Trigger |
|---------|------------|--------|-----|--------|---------|
| **IH** | 上证50 | `SHORT_AT_OPEN` | 2 手 | NDX=+2.07% | ≥ 2.0% |
| **IF** | 沪深300 | `SHORT_AT_OPEN` | 1 手 | NDX=+2.07% | ≥ 2.0% |
| **IM** | 中证1000 | `SHORT_AT_OPEN` | 1 手 | NDX=+2.07% | ≥ 1.0% |

**Total margin required:** 684,000 CNY (68.4% of 1,000,000)

### Execution plan
- **Entry:** 09:15-09:25 集合竞价挂卖单开空
- **Exit:** 14:57+ 集合竞价挂买单平今
- **Stop:** 硬止损 = 开仓价 × 1.005 (-0.5%)

### Full strategy table
| Rank | Strategy | Futures | US sig | US ret | Thr | Trig | Notes |
|------|----------|---------|--------|--------|-----|------|-------|
| 1 | IH_fade_short | IH | NDX | +2.07% | 2.0% | ✅ | 最稳; 10年里8年正收益; 大盘金融蓝筹反应最强 |
| 2 | IF_fade_short | IF | NDX | +2.07% | 2.0% | ✅ | 次稳; 与 IH 同信号但反应弱; |
| 3 | IC_fade_short | IC | SPX | +1.18% | 1.5% | ❌ | 用 SPX 阈值低; 2021-2023 连亏需警惕 |
| 4 | IM_fade_short | IM | NDX | +2.07% | 1.0% | ✅ | 样本仅5年; 2024年贡献全部收益; 高波动 |