# LLM Judge Drift Audit Report: 🟢 NO DRIFT (STABLE)
- **Timestamp**: `2026-10-31T18:06:59.235838Z`
- **Git Commit**: `4ea12e8`
- **Judge Model**: `mock-calibrated-70b` (mock)
- **Sample Size**: 15 golden probes
- **Overall Kappa**: Baseline = `0.000`, Current = `0.780` [95% CI: 0.040, 1.000]

## 1. Per-Axis Drift Breakdown (Bootstrap 95% CIs)
| Axis | Baseline Mean | Current Mean | Score Shift [95% CI] | Current Kappa [95% CI] | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fluency | 0.800 | 0.871 | +0.070 [-0.016, +0.124] | 0.878 [0.000, 1.000] | 🟢 OK |
| Coherence | 0.800 | 0.804 | +0.004 [-0.136, +0.102] | 0.539 [0.135, 1.000] | 🟢 OK |
| Instruction_Adherence | 0.800 | 0.834 | +0.034 [-0.086, +0.121] | 0.933 [0.000, 1.000] | 🟢 OK |
| No_Hallucination | 0.800 | 0.838 | +0.038 [-0.085, +0.138] | 0.769 [0.025, 1.000] | 🟢 OK |

## 2. Bias Monitoring
- Position Bias Rate: `0.04` (Shift detected: `False`)

## 3. Verdict Summary
All axes demonstrated alignment stability within 95% bootstrap confidence bounds.