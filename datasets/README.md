# Datasets

Place benchmark JSON / JSONL files here (or pass `--data_path`).

Bundled in this checkout when available:

| File | Benchmark | Notes |
|---|---|---|
| `hmmt.json` | HMMT 2025 Feb + Nov (60) | Math short answer |
| `gpqa_diamond.json` | GPQA-Diamond (198) | Multiple choice |
| `olympiad/test.jsonl` | OlympiadBench-Math subset | Math |
| `countdown6.jsonl` | Countdown (6 numbers) | Expression → target |
| `sgpqa200.json` | SuperGPQA 200-question subset | Used for Gemma in the paper |
| `aime.json` | AIME 2024+2025 (optional smoke) | |

`generate.py` resolves paths as:

1. `datasets/<name>` in this repo  
2. else `../Soft-Thinking/datasets/<name>` (local dev fallback)

Public sources:

- HMMT / MathArena-style releases  
- [GPQA](https://github.com/idavidrein/gpqa)  
- [OlympiadBench](https://github.com/OpenBMB/OlympiadBench)  
- SuperGPQA (select the 200-question PCB-hard subset used in the paper)
