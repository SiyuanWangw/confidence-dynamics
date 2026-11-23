# Confidence Dynamics for Test-Time Reasoning

Training-free test-time aggregation that jointly filters reasoning trajectories
by **tail confidence** \(C\) and **inter-layer confidence dynamics** \(\Delta C\),
then aggregates with \(\Delta C\)-weighted majority voting.

This repository reproduces the method from:

> *Beyond Final Confidence: Leveraging Confidence Dynamics for Test-Time Reasoning*
> (under review at ICLR 2027) — see [`docs/paper.pdf`](docs/paper.pdf).

## Method (Ours)

For each problem, sample \(N\) trajectories. For every trajectory \(\tau\):

1. **Token confidence** (DeepConf):  
   \(C_t = -\frac{1}{k}\sum_{j=1}^{k}\log p_t^{(L)}(j)\) with \(k=20\).
2. **Inter-layer dynamics** (default: penultimate → final):  
   \(\Delta C_t = C_t^{(L)} - C_t^{(L-1)}\).
3. Aggregate both over a **tail** window:
   - **Qwen3\***: last **2048** tokens  
   - **Gemma\***: last **20%** of the trajectory
4. **Joint filter** at retention ratio \(\rho\):  
   \(S_{C,\Delta}^{\rho} = \mathrm{Top}_{\rho}(C) \cap \mathrm{Top}_{\rho}(\Delta C)\)  
   Defaults: \(\rho=0.60\) (Qwen3), \(\rho=0.50\) (Gemma).
5. **\(\Delta C\)-weighted vote** over the retained set.

## Baselines

Evaluated on the **same** trajectory pool:

| Key | Method |
|---|---|
| `vote_acc` | Unweighted majority voting over all \(N\) trajectories |
| `vote_acc_tail_c_weighted` | Tail-confidence **weighted** majority voting |
| `vote_acc_tail_c_filter_20` | Keep top **20%** by tail \(C\), then unweighted MV |
| `vote_acc_tail_c_filter_30` | Keep top **30%** by tail \(C\), then unweighted MV |
| `vote_acc_tail_c_filter_50` | Keep top **50%** by tail \(C\), then unweighted MV |
| `vote_acc_joint_dc_weighted` | **Ours**: \(C@\rho \cap \Delta C@\rho\) + \(\Delta C\)-weighted MV |

## Repository layout

```
confidence-dynamics/
  aggregation.py          # Joint filter + voting (paper methods)
  filter_metrics.py       # Majority / weighted / top-k helpers
  generate.py             # Sample N trajectories + evaluate
  *_eval.py               # Dataset judges (HMMT, GPQA, Olympiad, …)
  configure.sh            # Install local SGLang + deps
  run_eval.sh             # Batch evaluation launcher
  sglang_pkg/             # SGLang v0.5.19 + C/ΔC layer-margin hooks
  datasets/               # Benchmark files (see datasets/README.md)
  tests/                  # CPU unit tests
  docs/paper.pdf          # Reference paper
```

`sglang_pkg/` is a vendored SGLang tree with hooks that expose per-token
final-layer \(C\) and penultimate–final \(\Delta C\) during decode
(`srt/layers/layer_margins.py`).

## Setup

```bash
# Python 3.12 recommended
conda create -n confdyn python=3.12 -y
conda activate confdyn

# Install torch (pick the CUDA index that matches your driver) + editable SGLang
bash configure.sh

python -c "import sglang; print(sglang.__file__, sglang.__version__)"
# should point at .../confidence-dynamics/sglang_pkg/python/sglang/...
```

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `OGF_LAYER_MARGINS` | `1` | Compute \(C\) / \(\Delta C\) during decode |
| `DEEPCONF_K` | `20` | Top-\(k\) for DeepConf confidence |
| `LOCAL_TAIL_WINDOW` | `2048` | Fixed tail length (Qwen) |
| `DUMP_STEP_STATS` | `1` | Write per-token sidecar `.steps.jsonl` |

## Quick start

```bash
# CPU tests (no GPU)
python -m pytest -q tests/test_aggregation.py tests/test_filter_metrics.py

# Smoke eval (place model weights under MODELS_DIR)
python generate.py \
  --model_path /path/to/Qwen3-1.7B \
  --dataset hmmt --end_idx 4 --num_samples 4 \
  --joint_rho 0.6 \
  --max_new_tokens 2048 \
  --dump_json results/hmmt_smoke.json
```

Batch runs:

```bash
# Qwen3 defaults: T=0.6 top_k=20, tail=2048, ρ=0.6
MODELS=Qwen3-8B DATASETS="hmmt gpqa_diamond olympiad countdown6" \
  NUM_SAMPLES=10 bash run_eval.sh

# Gemma4 defaults: T=1.0 top_k=64, tail=20%, ρ=0.5
MODELS=gemma-4-12B FAMILY=gemma4 DATASETS="hmmt gpqa_diamond olympiad sgpqa200" \
  NUM_SAMPLES=10 bash run_eval.sh
```

Paper decoding settings (Appendix D.2):

| Model | Dataset | T | top-k | top-p | max tokens |
|---|---|---|---|---|---|
| Qwen3-1.7B / 8B | HMMT | 0.6 | 20 | 0.95 | 40,000 |
| Qwen3-1.7B / 8B | GPQA / Olympiad | 0.6 | 20 | 0.95 | 32,768 |
| Qwen3-1.7B / 8B | Countdown | 1.0 | 20 | 0.95 | 32,768 |
| Gemma4-12B | HMMT | 1.0 | 64 | 0.95 | 40,000 |
| Gemma4-12B | others | 1.0 | 64 | 0.95 | 32,768 |

Truncated trajectories (hit max length) are excluded before aggregation, matching the paper.

## Offline re-aggregation

If you already have dumps with `paper_tail_c_list` / `paper_tail_dc_list`
(or `mean_tail_*_list`), you can recompute votes without regenerating:

```bash
python scripts/reaggregate.py results/hmmt_n10.json --rho 0.5
```

## Citation

```bibtex
@inproceedings{confidencedynamics2027,
  title     = {Beyond Final Confidence: Leveraging Confidence Dynamics for Test-Time Reasoning},
  author    = {Anonymous},
  booktitle = {ICLR},
  year      = {2027},
  note      = {Under review}
}
```

## License

- Project code: [MIT](LICENSE)
- `sglang_pkg/`: [Apache-2.0](sglang_pkg/LICENSE) (SGLang)
