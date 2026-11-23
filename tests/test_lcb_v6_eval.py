"""CPU checks for LiveCodeBench v6 (2025-02-01 through 2025-05-01)."""

import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.evals import lcb_eval
from confdyn.generate import (
    CODE_DATASETS,
    DEFAULT_LCB_V6,
    EVAL_DATASETS,
    _case_record,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _user_prompt,
)


def test_lcb_v6_is_registered():
    assert "lcb_v6" in CODE_DATASETS
    assert "lcb_v6" in EVAL_DATASETS
    assert _dataset_defaults("lcb_v6") == DEFAULT_LCB_V6


def test_lcb_v6_has_131_in_the_date_window():
    rows = _load_samples(DEFAULT_LCB_V6)
    assert len(rows) == 131
    assert rows[0]["id"] == "lcb_v6_000"
    assert rows[-1]["id"] == "lcb_v6_130"
    start = datetime(2025, 2, 1)
    end = datetime(2025, 5, 1)
    dates = [datetime.fromisoformat(r["contest_date"]) for r in rows]
    assert min(dates) >= start
    assert max(dates) <= end
    assert {r["platform"] for r in rows} == {"atcoder", "leetcode"}
    assert sum(1 for r in rows if (r.get("starter_code") or "").strip()) == 49
    assert len({r["question_id"] for r in rows}) == 131


def test_prompt_uses_official_format():
    rows = _load_samples(DEFAULT_LCB_V6)
    leet = next(r for r in rows if r["question_id"] == "3753")
    prompt = _user_prompt("lcb_v6", leet)
    assert prompt.startswith("You are an expert Python programmer.")
    assert "### Question:\n" in prompt
    assert "maxDifference" in prompt
    assert "```python\nclass Solution:" in prompt
    assert prompt.rstrip().endswith("### Answer: (use the provided format with backticks)")
    stdin = next(r for r in rows if not (r.get("starter_code") or "").strip())
    stdin_prompt = _user_prompt("lcb_v6", stdin)
    assert "stdout" in stdin_prompt
    assert "# YOUR CODE HERE" in stdin_prompt
    rec = _case_record("lcb_v6", leet)
    assert rec["question_id"] == "3753"
    assert "private_test_cases" not in rec


def test_extracts_last_python_block():
    text = "draft\n```python\nprint(1)\n```\nfinal\n```python\nprint(2)\n```\n"
    assert lcb_eval.extract_code(text) == "print(2)"


def test_stdin_and_functional_grader():
    ok, code = lcb_eval.rule_judge(
        "```python\nprint(sum(map(int, input().split())))\n```",
        {
            "public_test_cases": [
                {"input": "1 2\n", "output": "3\n", "testtype": "stdin"}
            ],
            "private_test_cases": [],
            "metadata": {},
        },
        True,
    )
    assert ok, code
    ok, _ = lcb_eval.rule_judge(
        "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```",
        {
            "public_test_cases": [{"input": "1\n2", "output": "3", "testtype": "functional"}],
            "private_test_cases": [],
            "metadata": {"func_name": "add"},
        },
        True,
    )
    assert ok
    ok, _ = lcb_eval.rule_judge(
        "```python\nprint(0)\n```",
        {
            "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
            "private_test_cases": [],
            "metadata": {},
        },
        True,
    )
    assert not ok


def test_real_leetcode_public_tests():
    rows = _load_samples(DEFAULT_LCB_V6)
    sample = next(r for r in rows if r["question_id"] == "3753")
    solution = """```python
class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        odd = max(v for v in freq.values() if v % 2 == 1)
        even = min(v for v in freq.values() if v % 2 == 0)
        return odd - even
```"""
    # Public cases only: the hidden set is large and this checks the call format.
    public = dict(sample)
    public["private_test_cases"] = "[]"
    ok, extracted = _judge_sample("lcb_v6", solution, public, True)
    assert ok, extracted
    ok, _ = _judge_sample("lcb_v6", "no code block", public, True)
    assert not ok
