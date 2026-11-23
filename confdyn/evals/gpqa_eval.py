"""GPQA / SuperGPQA prompt flattening and community-style letter extract.

Layout: simple-evals / EvalScope (stem + A) B) C) …).
Prompt: EvalScope ``ANSWER: [LETTER]`` last-line instruction.
Extract follows EvalScope / lm-eval flexible-extract / Artificial Analysis:
last match, post-think first, never math_verify / latex formulas.
GPQA-diamond uses A–D; SuperGPQA uses A–J (4–10 options).
"""

from __future__ import annotations

import re

LETTERS_ALL = "ABCDEFGHIJ"
DEFAULT_LETTERS = "ABCD"


def choice_letters(sample: dict | None = None) -> str:
    if not sample:
        return DEFAULT_LETTERS
    choices = sample.get("choices")
    if isinstance(choices, dict) and choices:
        keys = [k.upper() for k in choices if str(k).upper() in LETTERS_ALL]
        if keys:
            return "".join(sorted(keys, key=LETTERS_ALL.index))
    opts = sample.get("options")
    if isinstance(opts, list) and opts:
        return LETTERS_ALL[: len(opts)]
    n = sample.get("n_options")
    if isinstance(n, int) and 1 <= n <= len(LETTERS_ALL):
        return LETTERS_ALL[:n]
    return DEFAULT_LETTERS


def _letter_cls(letters: str) -> str:
    return f"[{letters}{letters.lower()}]"


def _extract_patterns(letters: str) -> tuple[re.Pattern[str], ...]:
    cls = _letter_cls(letters)
    json_pat = re.compile(
        rf"""["']answer["']\s*:\s*["']?\s*({cls})["']?""",
        re.IGNORECASE,
    )
    return (
        re.compile(rf"(?i)\bANSWER\s*:\s*\$?\(?\**({cls})\**\)?"),
        json_pat,
        re.compile(rf"\\boxed\{{[^}}]*({cls})[^}}]*\}}"),
        re.compile(rf"(?i)\bfinal answer is\s*:?\s*\(?({cls})\)?"),
        re.compile(rf"(?i)\bthe answer is\s*:?\s*\(?({cls})\)?"),
        re.compile(rf"(?i)\banswer is\s*:?\s*\(?({cls})\)?"),
        re.compile(rf"\(({cls})\)"),
        re.compile(rf"(?i)\b({cls})\s+is\s+the\s+correct\s+answer"),
    )


def query_template(letters: str = DEFAULT_LETTERS) -> str:
    joined = ", ".join(letters)
    return (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        f"'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {joined}. "
        "Think step by step before answering.\n\n"
        "{Question}"
    )


def _pop_trailing_options(lines: list[str], pattern: re.Pattern[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    while lines:
        if not lines[-1].strip():
            lines.pop()
            continue
        m = pattern.match(lines[-1].strip())
        if not m:
            break
        opts[m.group(1)] = m.group(2).strip()
        lines.pop()
    return opts


def _raw_question(sample: dict) -> str:
    prompt = sample.get("prompt")
    if isinstance(prompt, list) and prompt:
        return str(prompt[0]["value"])
    if isinstance(prompt, str):
        return prompt
    if sample.get("question"):
        return str(sample["question"])
    raise KeyError("gpqa sample is missing prompt/question")


def format_gpqa_question(sample: dict) -> str:
    """Stem + A) B) C) … with full option text (simple-evals / EvalScope)."""
    letters = choice_letters(sample)
    line_re = re.compile(rf"^([{letters}])[\.\)]\s+(.*)$")
    inner_re = re.compile(rf"^([{letters.lower()}])[\.\)]\s+(.*)$")
    lines = _raw_question(sample).splitlines()
    parsed = _pop_trailing_options(lines, line_re)
    choices = sample.get("choices") or {}
    texts: dict[str, str] = {}
    nested = True
    low = letters.lower()
    for letter in letters:
        val = str(choices.get(letter, parsed.get(letter, ""))).strip()
        if len(val) == 1 and val.lower() in low:
            texts[letter] = val.lower()
        else:
            nested = False
            texts[letter] = val or parsed.get(letter, "")
    if nested:
        inner = _pop_trailing_options(lines, inner_re)
        for letter in letters:
            key = texts[letter]
            texts[letter] = inner.get(key, parsed.get(letter, key))
    stem = "\n".join(lines).strip()
    opts = "\n".join(f"{letter}) {texts[letter]}" for letter in letters)
    return f"{stem}\n\n{opts}" if stem else opts


def _after_think(solution_str: str) -> str:
    for token in ("</think>", "<channel|>"):
        if token in solution_str:
            return solution_str.rsplit(token, 1)[-1]
    return solution_str


def _last_letter(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return str(matches[-1]).upper()


def _standalone_letter(text: str, letters: str) -> str | None:
    stripped = text.strip()
    last_line = re.compile(
        rf"^\s*(?:\*\*|`|'|\")?([{letters}])(?:\*\*|`|'|\")?(?:\.)?\s*$"
    )
    if len(stripped) == 1 and stripped.upper() in letters:
        return stripped.upper()
    for line in reversed(stripped.splitlines()):
        m = last_line.match(line)
        if m:
            return m.group(1).upper()
        if line.strip():
            break
    return None


def extract_choice(solution_str: str, letters: str = DEFAULT_LETTERS) -> str | None:
    """EvalScope / AA / lm-eval flexible letter. Prefers the post-think span."""
    spans = [_after_think(solution_str), solution_str]
    patterns = _extract_patterns(letters)
    for span in spans:
        if not span or not span.strip():
            continue
        for pattern in patterns:
            letter = _last_letter(pattern, span)
            if letter:
                return letter
        letter = _standalone_letter(span, letters)
        if letter:
            return letter
    return None


def extract_json_choice(solution_str: str, letters: str = DEFAULT_LETTERS) -> str | None:
    cls = _letter_cls(letters)
    pattern = re.compile(
        rf"""["']answer["']\s*:\s*["']?\s*({cls})["']?""",
        re.IGNORECASE,
    )
    matches = pattern.findall(solution_str)
    if not matches:
        return None
    return matches[-1].upper()


def rule_judge(
    solution_str: str,
    ground_truth: str,
    finish_generation: bool = True,
    letters: str = DEFAULT_LETTERS,
) -> tuple[bool, str]:
    del finish_generation
    gold_letter = str(ground_truth).strip().strip("\"'").upper()
    choice = extract_choice(solution_str, letters=letters)
    if not choice:
        return False, "No extracted answer"
    return choice == gold_letter, choice
