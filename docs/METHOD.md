# Method summary

## Signals

Per decode step \(t\):

\[
C_t^{(L)} = -\tfrac{1}{k}\sum_{j=1}^{k}\log p_t^{(L)}(j),\quad k=20
\]

\[
\Delta C_t = C_t^{(L)} - C_t^{(L-1)}
\]

Trajectory-level scores use the **tail** mean:

- Qwen3: last 2048 steps (`LOCAL_TAIL_WINDOW`)
- Gemma: last 20% of steps

## Aggregation (Ours)

\[
S_C^\rho=\mathrm{Top}_\rho(C),\quad
S_\Delta^\rho=\mathrm{Top}_\rho(\Delta C),\quad
S_{C,\Delta}^\rho = S_C^\rho \cap S_\Delta^\rho
\]

\[
\hat a = \arg\max_a \sum_{i\in S_{C,\Delta}^\rho} \Delta C(\tau_i)\, \mathbf{1}[a_i=a]
\]

Default \(\rho\): 0.6 (Qwen), 0.5 (Gemma).

## Code map

| Piece | File |
|---|---|
| DeepConf \(C\) / \(\Delta C\) at decode | `sglang_pkg/.../layer_margins.py` |
| Tail means | `filter_metrics.mean_tail_steps` / `mean_tail_frac` |
| Joint filter + vote | `aggregation.py` |
| End-to-end eval | `generate.py` |
| Offline re-vote | `scripts/reaggregate.py` |
