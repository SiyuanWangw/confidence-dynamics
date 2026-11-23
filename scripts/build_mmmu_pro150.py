#!/usr/bin/env python3
"""Build MMMU-Pro 150-item subset: 5 questions per subject (seed=42).

Prefers local parquet under ``datasets/mmmu_pro_raw/`` (including ModelScope
snapshot layout). Download with::

  python -c "from modelscope import dataset_snapshot_download as d; \\
    d('AI-ModelScope/MMMU_Pro', cache_dir='datasets/mmmu_pro_raw', \\
      allow_patterns=['standard (10 options)/*'])"

Falls back to streaming HuggingFace ``MMMU/MMMU_Pro`` if no local parquet.
Only rows with exactly 10 options are eligible. Writes:

  datasets/mmmu_pro150.json
  datasets/mmmu_pro150/images/*.png
"""

from __future__ import annotations

import ast
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT_DIR = DATA / "mmmu_pro150"
IMG_DIR = OUT_DIR / "images"
OUT_JSON = DATA / "mmmu_pro150.json"
RAW_DIR = DATA / "mmmu_pro_raw"
SEED = 42
PER_SUBJECT = 5
CONFIG = "standard (10 options)"
EXPECTED_BYTES = {
    "test-00000-of-00002.parquet": 346370445,
    "test-00001-of-00002.parquet": 335895602,
}


def _parse_options(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        # Some rows are double-encoded JSON/Python literals.
        for _ in range(3):
            parsed = None
            for loader in (ast.literal_eval, json.loads):
                try:
                    parsed = loader(s)
                    break
                except Exception:
                    continue
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            if isinstance(parsed, str):
                s = parsed.strip()
                continue
            break
    raise ValueError(f"bad options: {repr(raw)[:200]}")


def _decode_image(cell):
    if cell is None:
        return None
    if hasattr(cell, "save"):
        im = cell
        return im.convert("RGB") if getattr(im, "mode", None) != "RGB" else im
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = cell.get("path")
        if path and os.path.isfile(path):
            return Image.open(path).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    return None


def _local_parquets_ok() -> list[Path]:
    # Flat layout or ModelScope snapshot under mmmu_pro_raw/.../standard (10 options)/
    candidates = sorted(RAW_DIR.rglob("test-*-of-*.parquet"))
    # Prefer standard (10 options) when both 4/10 exist.
    ten = [p for p in candidates if "standard (10 options)" in str(p)]
    files = ten if ten else candidates
    ok = []
    seen_names: set[str] = set()
    for p in files:
        if p.name in seen_names:
            continue
        want = EXPECTED_BYTES.get(p.name)
        got = p.stat().st_size
        if want is not None and got != want:
            print(f"skip incomplete {p.name}: size={got} expected={want}", flush=True)
            continue
        seen_names.add(p.name)
        ok.append(p)
    return ok


def _iter_rows():
    local = _local_parquets_ok()
    if len(local) >= 2:
        import pyarrow as pa
        import pyarrow.parquet as pq

        print(f"Loading {len(local)} local parquet file(s)", flush=True)
        table = pa.concat_tables([pq.read_table(p) for p in local])
        rows = table.to_pylist()
        print(f"loaded {len(rows)} rows", flush=True)
        for r in rows:
            yield r
        return

    from datasets import load_dataset

    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}", flush=True)
    print(f"Streaming MMMU/MMMU_Pro {CONFIG!r} ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", CONFIG, split="test", streaming=True)
    for row in ds:
        yield row


def _reservoir_sample(rows_iter, k: int, seed: int):
    """Unbiased k-per-subject reservoir sample over a one-pass stream.

    Only keeps rows whose options parse to exactly 10 choices (the standard
    10-option split still contains some 2–9-option leftovers).
    """
    rng = random.Random(seed)
    picked: dict[str, list] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    skipped = 0
    for row in rows_iter:
        try:
            opts = _parse_options(row["options"])
        except Exception:
            skipped += 1
            continue
        if len(opts) != 10:
            skipped += 1
            continue
        # Cache parsed options so emit does not re-parse nested strings.
        row = dict(row)
        row["options"] = opts
        subj = str(row["subject"])
        seen[subj] += 1
        n = seen[subj]
        bucket = picked[subj]
        if len(bucket) < k:
            bucket.append(row)
        else:
            j = rng.randint(1, n)
            if j <= k:
                bucket[j - 1] = row
    print(f"reservoir: skipped {skipped} non-10-option rows", flush=True)
    return picked, seen


def _emit_sample(ex: dict, local_j: int, out_idx: int) -> dict:
    options = _parse_options(ex["options"])
    if len(options) != 10:
        raise RuntimeError(f"{ex.get('id')}: expected 10 options, got {len(options)}")
    ans = str(ex["answer"]).strip().upper()
    if ans not in "ABCDEFGHIJ":
        raise RuntimeError(f"{ex.get('id')}: bad answer {ans!r}")

    image_rels: list[str] = []
    for ki in range(1, 8):
        im = _decode_image(ex.get(f"image_{ki}"))
        if im is None:
            continue
        rel = f"mmmu_pro150/images/{ex['id']}_{ki}.png"
        path = DATA / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format="PNG")
        image_rels.append(rel)
    if not image_rels:
        raise RuntimeError(f"{ex.get('id')}: no images")

    choices = {chr(ord("A") + i): options[i] for i in range(10)}
    return {
        "id": f"mmmu_pro150_{out_idx:03d}",
        "mmmu_id": ex["id"],
        "subject": str(ex["subject"]),
        "source": "mmmu_pro_standard10",
        "kind": "mmmu_pro",
        "question": ex["question"],
        "options": options,
        "choices": choices,
        "final_answer": ans,
        "answer": ans,
        "n_options": 10,
        "images": image_rels,
        "img_type": ex.get("img_type"),
        "topic_difficulty": ex.get("topic_difficulty"),
        "explanation": ex.get("explanation"),
        "sample_seed": SEED,
        "subject_local_idx": local_j,
    }


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    picked, seen = _reservoir_sample(_iter_rows(), PER_SUBJECT, SEED)
    if len(picked) != 30:
        raise RuntimeError(f"expected 30 subjects, got {len(picked)}: {sorted(picked)}")
    for subj, bucket in picked.items():
        if len(bucket) != PER_SUBJECT:
            raise RuntimeError(f"{subj}: got {len(bucket)} (seen={seen[subj]})")

    rows_out = []
    for subj in sorted(picked):
        # Stable order within subject by mmmu id after reservoir picks.
        bucket = sorted(picked[subj], key=lambda r: str(r["id"]))
        for local_j, ex in enumerate(bucket):
            rows_out.append(_emit_sample(ex, local_j, len(rows_out)))
            if len(rows_out) % 10 == 0:
                print(f"  wrote {len(rows_out)}/150", flush=True)

    OUT_JSON.write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(r["subject"] for r in rows_out)
    print(f"wrote {OUT_JSON} ({len(rows_out)} rows)", flush=True)
    print(
        f"images under {IMG_DIR} ({sum(1 for _ in IMG_DIR.glob('*.png'))} files)",
        flush=True,
    )
    assert len(rows_out) == 30 * PER_SUBJECT
    assert all(v == PER_SUBJECT for v in counts.values())
    return 0


if __name__ == "__main__":
    sys.exit(main())
