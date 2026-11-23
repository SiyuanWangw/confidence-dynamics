"""CPU checks for EvalScope / lm-eval / AA GPQA prompt and letter extract."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from generate import DEFAULT_GPQA_DIAMOND, GPQA_QUERY_TEMPLATE, MCQ_DATASETS, _load_samples, _user_prompt
from gpqa_eval import extract_choice, extract_json_choice, format_gpqa_question, rule_judge

_SAMPLE = {
    "prompt": [
        {
            "from": "user",
            "value": (
                "Which planet is densest?\n\n"
                "a) Earth analog.\n"
                "b) 2 Earth masses.\n"
                "c) 5x Earth mass.\n"
                "d) half Earth mass.\n\n"
                "A. a\nB. b\nC. c\nD. d"
            ),
        }
    ],
    "final_answer": "C",
    "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
}


def test_gpqa_is_mcq():
    assert "gpqa_diamond" in MCQ_DATASETS


def test_nested_options_flattened_to_simple_evals():
    q = format_gpqa_question(_SAMPLE)
    assert q.startswith("Which planet is densest?")
    assert "C) 5x Earth mass." in q
    assert "A. a" not in q
    prompt = _user_prompt("gpqa_diamond", _SAMPLE)
    assert "ANSWER: [LETTER]" in prompt
    assert prompt.startswith("Answer the following multiple choice question.")
    assert q in prompt
    assert 'e.g., "answer": "C".' not in prompt
    assert "Please solve the following multiple-choice question" not in prompt
    assert "{Question}" in GPQA_QUERY_TEMPLATE


def test_full_choices_use_A_paren():
    sample = {
        "prompt": [
            {
                "from": "user",
                "value": "Q?\n\nA. Warm\nB. Cold molecular\nC. Cold atomic\nD. Warm molecular",
            }
        ],
        "final_answer": "C",
        "choices": {
            "A": "Warm",
            "B": "Cold molecular",
            "C": "Cold atomic",
            "D": "Warm molecular",
        },
    }
    q = format_gpqa_question(sample)
    assert q == "Q?\n\nA) Warm\nB) Cold molecular\nC) Cold atomic\nD) Warm molecular"


def test_first_dataset_item_unnests():
    rows = _load_samples(DEFAULT_GPQA_DIAMOND)
    assert len(rows) == 198
    q = format_gpqa_question(rows[0])
    assert "A) An Earth-mass and Earth-radius planet." in q
    assert "C) A planet with the same composition as Earth but 5 times more massive than Earth." in q
    assert "A. a" not in q
    prompt = _user_prompt("gpqa_diamond", rows[0])
    assert q in prompt
    assert "ANSWER: [LETTER]" in prompt


def test_evalscope_answer_line():
    ok, extracted = rule_judge("lots of A in the stem.\nANSWER: D", "D")
    assert ok
    assert extracted == "D"


def test_answer_line_self_correction():
    ok, extracted = rule_judge("ANSWER: A\nwait\nANSWER: C", "C")
    assert ok
    assert extracted == "C"


def test_json_answer_still_ok():
    text = 'Reasoning mentions A.\n{"answer": "C"}'
    ok, extracted = rule_judge(text, "C")
    assert ok
    assert extracted == "C"
    assert extract_json_choice(text) == "C"
    assert extract_choice(text) == "C"


def test_json_wrong_letter():
    ok, extracted = rule_judge('"answer": "A"', "C")
    assert not ok
    assert extracted == "A"


def test_boxed_letter():
    ok, extracted = rule_judge(r"Reasoning. \boxed{C}", "C")
    assert ok, extracted
    assert extracted == "C"


def test_boxed_text_letter():
    ok, extracted = rule_judge(r"done \boxed{\text{C}}", "C")
    assert ok, extracted
    assert extracted == "C"


def test_the_answer_is():
    ok, extracted = rule_judge("I eliminate A and B.\nThe answer is C", "C")
    assert ok
    assert extracted == "C"


def test_lm_eval_paren_last():
    ok, extracted = rule_judge("maybe (A) then later (C)", "C")
    assert ok
    assert extracted == "C"


def test_post_think_preferred():
    text = "<think>The answer is A</think>\nANSWER: C"
    ok, extracted = rule_judge(text, "C")
    assert ok
    assert extracted == "C"


def test_post_think_preferred_gemma4():
    text = "<|channel>thought\nThe answer is A<channel|>\nANSWER: C"
    ok, extracted = rule_judge(text, "C")
    assert ok
    assert extracted == "C"


def test_latex_garbage_is_not_an_answer():
    ok, extracted = rule_judge("[rho_d < 5.5, '< 5.5']", "C")
    assert not ok
    assert extracted == "No extracted answer"
