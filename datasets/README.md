# Datasets for Table 1

`python -m confdyn` loads these files from this directory.

| File | Benchmark | Size | Task |
|---|---|---|---|
| `gpqa_diamond.json` | GPQA-Diamond | 198 | Multiple choice |
| `hmmt.json` | HMMT 2025 February and November | 60 | Short answer |
| `aime.json` | AIME 2024 + 2025 + 2026 | 90 | Integer in \([0, 999]\) |
| `sgpqa200.json` | SuperGPQA hard subset | 200 | 4–10 choices. 110 physics, 70 chemistry, 20 biology; 156 questions have ten choices |
| `countdown6_all.jsonl` | Countdown6 | 200 | Six numbers, each used once, to reach a target |

Public sources:

- [GPQA](https://github.com/idavidrein/gpqa)
- HMMT via the MathArena releases
- AIME 2024–2026
- [SuperGPQA](https://github.com/SuperGPQA/SuperGPQA): the 200-question subset in `sgpqa200.json`
- Countdown6: the 200 filtered use-all-six problems in `countdown6_all.jsonl`
