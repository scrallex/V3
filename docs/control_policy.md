# Control Policy

The control layer follows the **advisory-first** approach: provide clear human-readable
recommendations, then allow guarded auto-tuning once improvements are proven (ABA study).

## 1. Advisory Mode

Every scored window generates at most two suggestions (`scripts/telemetry/advisory.py`):

| Scenario | Action | Notes |
| --- | --- | --- |
| `sep_score < threshold` | Decrease power limit by 5 W | Helps recover GH/J; re-evaluate after 2 minutes. |
| `reject_rate > 0.02` | Suggest pool switch | Advisory only (delta = `None`). |
| `thermal_headroom < 3 °C` | Increase fan curve by 10 % | TTL 180 s to allow cool-down. |
| `rupture_flag == macro` | Reduce core clock by 15 MHz | TTL equals rupture cooldown (default 600 s). |
| `rupture_flag` transitions `none → micro` | Temporary fan boost | Pre-emptive response to emerging instability. |

Advisories are stored under `telemetry:advice:{rig}` (Valkey) and appended to
`logs/telemetry/<rig>_advice.ndjson`.

## 2. Auto-Tune Mode (future switch)

Once advisory effectiveness is confirmed:

1. **Actuator gating:** never change more than one actuator within a 2‑minute window.
2. **Rollback:** revert changes if efficiency drops by >2 % or reject rate increases by
   >0.5 pp within 3 minutes.
3. **Cooldown:** enforce a 10‑minute quiet period after any macro rupture.
4. **State tracking:** maintain per-actuator state machine (suggested module:
   `scripts/telemetry/autotune.py`, to be implemented) using advisory outputs as intent.

## 3. Safety Rails

- Read kill switch from `ops:kill_switch`. The control loop must abort if trading mode is
  disabled.
- Validate target ranges against rig metadata (`config/telemetry.yaml`), e.g. never drop
  power limit below PSU stability or raise fan curve above vendor max.
- Log every attempted change (idempotent) with timestamp, actuator, delta, and observed
  metrics before/after; required for ABA replay.

## 4. GPU Considerations (RTX 3080 Ti)

- NVML-based power/clock adjustments require administrative privileges and may reset on
  reboot. Scripts should validate support via `pynvml.nvmlDeviceGetPowerManagementMode`.
- Cooling adjustments should respect board partner limits (most 3080 Ti cards allow up to
  110 % fan speed).
- Use burst sampling to capture transient thermal spikes before deciding on aggressive
  tuning.

## 5. Escalation to Full Control

1. Finish advisory ABA run for at least one rig (Week 2 milestone).
2. Enable guarded power-limit adjustments with rollback (Week 3).
3. Expand to multi-actuator policy with bandit tuning (Week 4+).

Until step 2, keep the auto-tune module disabled and rely solely on the advisory channel.
