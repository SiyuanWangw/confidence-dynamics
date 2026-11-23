"""OlympiadBench rule judge (boxed LaTeX via math_verify)."""

from __future__ import annotations

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


def rule_judge(
    solution_str: str, ground_truth: str, finish_generation: bool = True
) -> tuple[bool, str]:
    del finish_generation
    gt = str(ground_truth)
    if not gt.startswith("$"):
        gt = f"${gt}$"
    gold = parse(gt, extraction_config=[LatexExtractionConfig()])
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    return bool(verify(gold, answer)), str(answer)
