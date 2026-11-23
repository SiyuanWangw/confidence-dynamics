"""Dataset paths, prompts, and answer judging."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

from confdyn.evals import (
    aime_eval,
    countdown_eval,
    gpqa_eval,
    hle_eval,
    hmmt_eval,
    lcb_eval,
    mmmu_eval,
    olympiad_eval,
)
from confdyn.prompting import _strip_think

COUNTDOWN_TEMPLATE = """
{Question}
You may use basic arithmetic operations (+, -, *, /), and each number can only be used once. Please reason step by step and show your reasoning inside <think> </think> tags. Then provide only the final equation enclosed in \\boxed{{}}, for example: \\boxed{{(1 + 2) / 3}}.
"""
COUNTDOWN_TEMPLATE_GEMMA4 = """
{Question}
You may use basic arithmetic operations (+, -, *, /), and each number can only be used once. Please reason step by step. Then provide only the final equation enclosed in \\boxed{{}}, for example: \\boxed{{(1 + 2) / 3}}.
"""

MATH_QUERY_TEMPLATE = (
    "{Question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
GPQA_QUERY_TEMPLATE = gpqa_eval.query_template()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_DATA = os.path.join(_REPO_ROOT, "datasets")


def _data_path(*parts: str) -> str:
    """Resolve a path under this repo's ``datasets/`` directory."""
    return os.path.join(_LOCAL_DATA, *parts)


DEFAULT_COUNTDOWN6 = _data_path("countdown6.jsonl")
DEFAULT_COUNTDOWN6_ALL = _data_path("countdown6_all.jsonl")
DEFAULT_OLYMPIAD = _data_path("olympiad", "test.jsonl")
DEFAULT_HMMT = _data_path("hmmt.json")
DEFAULT_AIME = _data_path("aime.json")
DEFAULT_GSM8K = _data_path("gsm8k.json")
DEFAULT_GPQA_DIAMOND = _data_path("gpqa_diamond.json")
DEFAULT_SGPQA200 = _data_path("sgpqa200.json")
DEFAULT_HLE300 = _data_path("hle300.json")
DEFAULT_MMMU_PRO150 = _data_path("mmmu_pro150.json")
DEFAULT_LCB_V6 = _data_path("lcb_v6.json")
# aime is AIME 2024 + 2025 + 2026 (90).
# hle300 is HLE text-only short answers: 100 int + 100 number + 100 word.
# mmmu_pro150: MMMU-Pro standard (10 options), 5/subject × 30 = 150 (multimodal).
# lcb_v6: LiveCodeBench release_v6, contest dates 2025-02-01 through 2025-05-01 (131).
MATH_QA_DATASETS = ("olympiad", "hmmt", "aime", "gsm8k", "hle300")
MCQ_DATASETS = ("gpqa_diamond", "sgpqa200")
VLM_MCQ_DATASETS = ("mmmu_pro150",)
COUNTDOWN_DATASETS = ("countdown6", "countdown6all")
CODE_DATASETS = ("lcb_v6",)
EVAL_DATASETS = (
    COUNTDOWN_DATASETS + MATH_QA_DATASETS + MCQ_DATASETS + VLM_MCQ_DATASETS + CODE_DATASETS
)

def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_samples(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"expected a JSON list in {path}")

def _dataset_defaults(name: str) -> str:
    if name == "olympiad":
        return DEFAULT_OLYMPIAD
    if name == "hmmt":
        return DEFAULT_HMMT
    if name == "aime":
        return DEFAULT_AIME
    if name == "gsm8k":
        return DEFAULT_GSM8K
    if name == "gpqa_diamond":
        return DEFAULT_GPQA_DIAMOND
    if name == "sgpqa200":
        return DEFAULT_SGPQA200
    if name == "hle300":
        return DEFAULT_HLE300
    if name == "mmmu_pro150":
        return DEFAULT_MMMU_PRO150
    if name == "lcb_v6":
        return DEFAULT_LCB_V6
    if name == "countdown6":
        return DEFAULT_COUNTDOWN6
    if name == "countdown6all":
        return DEFAULT_COUNTDOWN6_ALL
    raise ValueError(f"unknown dataset {name}")


def _math_question(sample: dict) -> str:
    if sample.get("question"):
        return sample["question"]
    prompt = sample.get("prompt")
    if isinstance(prompt, list) and prompt:
        return prompt[0]["value"]
    if isinstance(prompt, str):
        return prompt
    raise KeyError("sample is missing question/prompt")


def _math_answer(sample: dict) -> str:
    if sample.get("answer") is not None:
        return str(sample["answer"])
    return str(sample["final_answer"])


def _user_prompt(dataset: str, sample: dict, family: str = "") -> str:
    if dataset in VLM_MCQ_DATASETS:
        return mmmu_eval.query_text(sample)
    if dataset in CODE_DATASETS:
        return lcb_eval.query_text(sample)
    if dataset in MCQ_DATASETS:
        letters = gpqa_eval.choice_letters(sample)
        return gpqa_eval.query_template(letters).format(
            Question=gpqa_eval.format_gpqa_question(sample)
        )
    if dataset in MATH_QA_DATASETS:
        return MATH_QUERY_TEMPLATE.format(Question=_math_question(sample))
    countdown = COUNTDOWN_TEMPLATE_GEMMA4 if family == "gemma4" else COUNTDOWN_TEMPLATE
    return countdown.format(Question=sample["prompt"])


def _chat_prompt_and_images(
    dataset: str, sample: dict, tokenizer, family: str
) -> Tuple[str, List[str] | None]:
    """Chat-templated prompt plus optional absolute image paths for Engine.generate."""
    from confdyn.prompting import _apply_chat

    if dataset in VLM_MCQ_DATASETS:
        content, images = mmmu_eval.build_user_content(sample)
        return _apply_chat(tokenizer, content, family), images
    return _apply_chat(tokenizer, _user_prompt(dataset, sample, family), family), None


def _judge_sample(
    dataset: str, text: str, sample: dict, finished: bool, family: str = ""
) -> Tuple[bool, str]:
    text = _strip_think(text, family)
    if dataset == "olympiad":
        return olympiad_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset == "hmmt":
        return hmmt_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset in ("aime", "gsm8k"):
        return aime_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset == "hle300":
        return hle_eval.rule_judge(text, sample, finished)
    if dataset in VLM_MCQ_DATASETS:
        return mmmu_eval.rule_judge(text, sample, finished)
    if dataset in MCQ_DATASETS:
        return gpqa_eval.rule_judge(
            text,
            sample["final_answer"],
            finished,
            letters=gpqa_eval.choice_letters(sample),
        )
    if dataset in COUNTDOWN_DATASETS:
        return countdown_eval.rule_judge(
            text, int(sample["target"]), list(sample["nums"]), variant=dataset
        )
    if dataset in CODE_DATASETS:
        return lcb_eval.rule_judge(text, sample, finished)
    raise ValueError(f"unknown dataset {dataset}")


def _case_record(dataset: str, sample: dict) -> dict:
    if dataset in VLM_MCQ_DATASETS:
        rec = {
            "prompt": mmmu_eval.format_question(sample),
            "ground_truth": sample["final_answer"],
            "choices": sample.get("choices"),
            "images": sample.get("images"),
            "id": sample.get("id"),
            "mmmu_id": sample.get("mmmu_id"),
            "subject": sample.get("subject"),
        }
        return rec
    if dataset in MCQ_DATASETS:
        rec = {
            "prompt": gpqa_eval.format_gpqa_question(sample),
            "ground_truth": sample["final_answer"],
            "choices": sample.get("choices"),
        }
        if sample.get("id"):
            rec["id"] = sample["id"]
        if sample.get("field"):
            rec["field"] = sample["field"]
        return rec
    if dataset in MATH_QA_DATASETS:
        rec = {
            "prompt": _math_question(sample),
            "ground_truth": _math_answer(sample),
            "id": sample.get("id"),
        }
        if sample.get("source"):
            rec["source"] = sample["source"]
        if sample.get("kind"):
            rec["kind"] = sample["kind"]
        return rec
    if dataset in CODE_DATASETS:
        return {
            "prompt": sample.get("question_title") or sample.get("question_id"),
            "ground_truth": sample.get("question_id"),
            "id": sample.get("id"),
            "question_id": sample.get("question_id"),
            "platform": sample.get("platform"),
            "difficulty": sample.get("difficulty"),
            "contest_date": sample.get("contest_date"),
        }
    return {
        "prompt": sample["prompt"],
        "ground_truth": sample["target"],
        "nums": sample["nums"],
        "answer": sample.get("answer"),
    }

