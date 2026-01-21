# Game Plan: Unbounded "Stream" Entry Strategy

## Objective
Shift from "Safe Stacking" (3 positions, 5% each) to "Stream Entry" (Unlimited positions, 0.5% each).
This allows the system to enter *every single time* the signal refreshes (approx. every 1 minute candles), creating a weighted average entry into trending moves.

## 1. Safety & Sizing Changes
**Current:**
- Risk per Trade: 5.0% (`0.05`)
- Max Positions: 3
- Entry Frequency: Capped by Stack Limit (Effective pauses)

**Proposed:**
- Risk per Trade: **0.5% (`0.005`)**
- Max Positions: **Unlimited (Set to 500)**
- Entry Frequency: Every Candle Update (~1 min) while Signal is valid.

## 2. Implementation Steps

### Step A: Shrink the Bullets (Crucial First Step)
We must lower the sizing *before* uncapping the limit to prevent massive over-leverage.
- **File:** `docker-compose.hotband.yml`
- **Action:** Set `PORTFOLIO_NAV_RISK_PCT` to `0.005`.

### Step B: Remove the Jam (Uncap)
Remove the artificial "3 bullet" limit.
- **File:** `config/live_config_safe.yaml`
- **Action:** Set `max_positions` to `500` (effectively simple infinite for a 6h horizon).

### Step C: Verification
- **Safety Check:** Ensure `ts_ms` (timestamp) updates on a Candle basis (1 min), not a Cycle basis (2 sec).
- **Result:** You will see 1 trade enter every ~1 minute as long as the signal is green.
- **Exposure:** In 1 hour of trend, you will accumulate ~60 small trades (30% NAV). This is aggressive but "calibrated" by the 0.5% sizing.

## 3. Why this works
- **Individual Timers:** Each of the 60 trades has its own 6-hour clock. They will exit sequentially 6 hours later, smoothing out the exit curve just like the entry curve.
- **No Pauses:** The system never "sits and waits". It effectively Dollar-Cost-Averages (DCA) into the trend.

## Execution
Reply **"EXECUTE"** to apply these changes immediately.
