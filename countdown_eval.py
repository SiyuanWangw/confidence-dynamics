"""Countdown rule judges.

countdown6 (Soft-Thinking COUNTDOWN6Evaluator): boxed equation uses exactly 4
of the 6 numbers, with 2 from nums[:3] and 2 from nums[3:].

countdown5: boxed equation uses each of the 5 given numbers exactly once.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction
from itertools import combinations


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string: str) -> str:
    string = str(string).strip().replace("\n", "").rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = string.replace("\\times", "*").replace("\\cdot", "*")
    string = string.replace(" ", "")
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def extract_answer_cal(pred_str: str, use_last_number: bool = True) -> str:
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            pred = ""
        elif ans[0] == "{":
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
            pred = a
        else:
            pred = ans.split("$")[0].strip()
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    else:
        if use_last_number:
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


def _extract_equation_text(answer_equation: str) -> str:
    text = (answer_equation or "").strip()
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1).strip()
    if "=" in text:
        text = text.split("=")[-1].strip()
    return text


def _extract_braced(text: str, start_idx: int) -> tuple[str, int]:
    if start_idx >= len(text) or text[start_idx] != "{":
        raise ValueError("expected '{'")
    depth = 0
    i = start_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : i], i + 1
        i += 1
    raise ValueError("unbalanced braces")


def _normalize_equation(expr: str) -> str:
    normalized = expr.strip()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\times", "*").replace("\\cdot", "*")
    normalized = normalized.replace("\\div", "/")
    while "\\frac" in normalized:
        frac_idx = normalized.find("\\frac")
        i = frac_idx + len("\\frac")
        while i < len(normalized) and normalized[i].isspace():
            i += 1
        numerator, next_idx = _extract_braced(normalized, i)
        while next_idx < len(normalized) and normalized[next_idx].isspace():
            next_idx += 1
        denominator, end_idx = _extract_braced(normalized, next_idx)
        replacement = f"(({numerator})/({denominator}))"
        normalized = normalized[:frac_idx] + replacement + normalized[end_idx:]
    return normalized.replace("{", "(").replace("}", ")")


def _eval_expr_and_collect_nums(expr: str) -> tuple[Fraction, list[int]]:
    tree = ast.parse(expr, mode="eval")
    used_nums: list[int] = []

    def _visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.BinOp):
            left, right = _visit(node.left), _visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            raise ValueError("unsupported operator")
        if isinstance(node, ast.UnaryOp):
            raise ValueError("unary operators are not allowed")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError("only integer constants are allowed")
            used_nums.append(node.value)
            return Fraction(node.value, 1)
        raise ValueError("unsupported expression")

    return _visit(tree), used_nums


def verify_use_all_equation(given_num: list[int], target: int, answer_equation: str) -> bool:
    """Countdown-5: use every given number exactly once; result equals target."""
    try:
        expr = _normalize_equation(_extract_equation_text(answer_equation))
        if not expr:
            return False
        result, used_nums = _eval_expr_and_collect_nums(expr)
        if Counter(used_nums) != Counter(int(x) for x in given_num):
            return False
        return result == Fraction(target, 1)
    except Exception:
        return False


def verify_partial_equation(given_num: list[int], target: int, answer_equation: str) -> bool:
    try:
        expr = _normalize_equation(_extract_equation_text(answer_equation))
        if not expr:
            return False
        result, used_nums = _eval_expr_and_collect_nums(expr)
        if len(given_num) != 6 or len(used_nums) != 4:
            return False
        expected_counter = Counter(given_num)
        used_counter = Counter(used_nums)
        if any(used_counter[x] > expected_counter[x] for x in used_counter):
            return False
        first_counter = Counter(given_num[:3])
        last_counter = Counter(given_num[3:])
        for idxs in combinations(range(4), 2):
            first_pick = [used_nums[i] for i in idxs]
            second_pick = [used_nums[i] for i in range(4) if i not in idxs]
            c1, c2 = Counter(first_pick), Counter(second_pick)
            if all(c1[x] <= first_counter[x] for x in c1) and all(
                c2[x] <= last_counter[x] for x in c2
            ):
                return result == Fraction(target, 1)
        return False
    except Exception:
        return False


def rule_judge(
    solution_str: str,
    target: int,
    given_num: list,
    variant: str = "countdown6",
) -> tuple[bool, str]:
    prediction = extract_answer_cal(solution_str)
    answer = strip_string(prediction).split("=")[0].strip()
    if len(answer) == 0:
        return False, "No extracted answer"
    if variant == "countdown5":
        ok = verify_use_all_equation(given_num, target, answer)
    else:
        ok = verify_partial_equation(given_num, target, answer)
    return ok, str(answer)
