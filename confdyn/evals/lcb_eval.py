"""LiveCodeBench v6 (2025-02-01 .. 2025-05-01) prompt and judge.

The prompt is the official code-generation template: the generic system
line plus the question, starter code when the problem has one, and a
python fence. Thinking models receive that as the user turn so the chat
template can still enable thinking.

A trace passes only when the last fenced block passes every public and
private test. Grading runs in a child process.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import zlib

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)
TEST_TIMEOUT = 6
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def query_text(sample: dict) -> str:
    prompt = f"{SYSTEM_MESSAGE}\n\n### Question:\n{sample['question_content']}\n\n"
    starter = sample.get("starter_code") or ""
    if starter:
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(text: str) -> str:
    """Last fenced block. The language tag stays on the opening fence line."""
    lines = (text or "").split("\n")
    idxs = [i for i, line in enumerate(lines) if "```" in line]
    if len(idxs) < 2:
        return ""
    return "\n".join(lines[idxs[-2] + 1 : idxs[-1]])


def _decode_tests(raw) -> list:
    if isinstance(raw, list):
        return raw
    text = raw or "[]"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
        )
    if isinstance(data, str):
        data = json.loads(data)
    return data


def tests_for(sample: dict) -> dict:
    cached = sample.get("_lcb_tests")
    if cached is not None:
        return cached
    meta = sample.get("metadata") or "{}"
    if isinstance(meta, str):
        meta = json.loads(meta)
    public = _decode_tests(sample.get("public_test_cases") or "[]")
    private = _decode_tests(sample.get("private_test_cases") or "[]")
    tests = public + private
    packed = {
        "fn_name": meta.get("func_name") or None,
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
    }
    sample["_lcb_tests"] = packed
    return packed


def rule_judge(text: str, sample: dict, finished: bool = True) -> tuple[bool, str]:
    """Score the extracted program. ``finished`` does not force a failure.

    A length-capped trace can still contain a complete program. Tree search
    separately marks an unfinished kept trace incorrect.
    """
    del finished
    code = extract_code(text)
    if not code.strip():
        return False, ""
    tests = tests_for(sample)
    if not tests["inputs"]:
        return False, code
    payload = {
        "code": code,
        "fn_name": tests["fn_name"],
        "inputs": tests["inputs"],
        "outputs": tests["outputs"],
        "timeout": TEST_TIMEOUT,
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    n_tests = len(tests["inputs"])
    wall = TEST_TIMEOUT * max(n_tests, 1) + 20
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "confdyn.evals.lcb_sandbox"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=wall,
            env=env,
            cwd=_REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return False, code
    if proc.returncode != 0:
        return False, code
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return False, code
    try:
        result = json.loads(line[-1])
    except json.JSONDecodeError:
        return False, code
    return bool(result.get("ok")), code
