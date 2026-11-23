"""GSM8K / gsm8k_p2 rule judge, matching Soft-Thinking matheval.GSM8KEvaluator."""

from __future__ import annotations

import re

from math_verify import ExprExtractionConfig, parse, verify

from countdown_eval import strip_string


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    elif use_last_number:
        found = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
        pred = found[-1] if found else ""
    else:
        pred = ""

    pred = re.sub(r"\n\s*", "", pred)
    if pred.startswith(":"):
        pred = pred[1:]
    if pred.endswith("."):
        pred = pred[:-1]
    if pred.endswith("/"):
        pred = pred[:-1]
    return strip_string(pred)


def rule_judge(
    solution_str: str, ground_truth: str, finish_generation: bool = True
) -> tuple[bool, str]:
    del finish_generation
    gold = parse(str(ground_truth), extraction_config=[ExprExtractionConfig()])
    answer = extract_answer(solution_str)
    if len(answer) == 0:
        return False, "No extracted answer"
    return bool(verify(gold, answer)), str(answer)
