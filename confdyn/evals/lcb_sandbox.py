"""Grade one LiveCodeBench solution in a fresh process.

The parent must not exec model code: the official grader calls
``reliability_guard``, which disables ``os`` and ``subprocess`` in-process.
This module is the child. It follows LiveCodeBench ``testing_util`` for
stdin and call-based problems (6s alarm per test, early exit on the first
failure, decimal line compare for stdio).
"""

from __future__ import annotations

import ast
import faulthandler
import json
import platform
import signal
import sys
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\n"
    "import bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
    "import itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def truncatefn(s, length=300):
    s = s if isinstance(s, str) else str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)
        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return (
            IMPORT_STRING
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str):
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code, all_inputs, all_outputs, fn_name, timeout):
    code = IMPORT_STRING + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None or not hasattr(compiled_sol, fn_name):
        return None
    method = getattr(compiled_sol, fn_name)
    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]
    total_execution = 0
    all_results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as exc:
            signal.alarm(0)
            timed_out = "timeoutexception" in repr(exc).lower()
            all_results.append(-3 if timed_out else -4)
            return all_results, {
                "error": repr(exc),
                "error_code": -3 if timed_out else -4,
                "error_message": "Time Limit Exceeded" if timed_out else "Runtime Error",
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return all_results, {"execution time": total_execution}


def grade_stdio(code, all_inputs, all_outputs, timeout):
    code = make_function(clean_if_name(code))
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None or not hasattr(compiled_sol, "wrapped_function"):
        return None
    method = getattr(compiled_sol, "wrapped_function")
    all_results = []
    total_execution_time = 0
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                timed_out = "timeoutexception" in repr(exc).lower()
                all_results.append(-3 if timed_out else -4)
                return all_results, {
                    "error": repr(exc),
                    "error_code": -3 if timed_out else -4,
                    "error_message": "Time Limit Exceeded" if timed_out else "Runtime Error",
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()
        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            return all_results, {"error_code": -2, "error_message": "Wrong answer: mismatched output length"}
        for output_line_idx, (pred_line, gt_line) in enumerate(
            zip(stripped_prediction_lines, stripped_gt_out_lines)
        ):
            if pred_line == gt_line:
                continue
            ok_pred, decimal_pred = convert_line_to_decimals(pred_line)
            ok_gt, decimal_gt = convert_line_to_decimals(gt_line)
            if ok_pred and ok_gt and decimal_pred == decimal_gt:
                continue
            all_results.append(-2)
            return all_results, {
                "error_code": -2,
                "error_message": f"Wrong answer at {output_line_idx}",
            }
        all_results.append(True)
    return all_results, {"execution time": total_execution_time}


def reliability_guard(maximum_memory_bytes=None):
    """Limit the child. This is not a security sandbox."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    faulthandler.disable()
    import builtins
    import os
    import shutil
    import subprocess

    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.chdir = None
    shutil.rmtree = None
    shutil.move = None
    subprocess.Popen = None


def grade_payload(payload: dict) -> dict:
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        reliability_guard(8 * 1024**3)
    except Exception:
        signal.signal(signal.SIGALRM, timeout_handler)
    code = payload.get("code") or ""
    fn_name = payload.get("fn_name") or None
    inputs = payload.get("inputs") or []
    outputs = payload.get("outputs") or []
    timeout = int(payload.get("timeout") or 6)
    try:
        if fn_name:
            graded = grade_call_based(code, inputs, outputs, fn_name, timeout)
        else:
            graded = grade_stdio(code, inputs, outputs, timeout)
    except Exception as exc:
        return {"ok": False, "n_passed": 0, "error": f"Error during testing: {exc}"}
    if not graded:
        return {"ok": False, "n_passed": 0, "error": "compile error"}
    results, meta = graded
    n_passed = sum(1 for item in results if item is True)
    ok = bool(results) and n_passed == len(inputs) and len(results) == len(inputs)
    error = "" if ok else str(meta.get("error_message") or meta.get("error") or "fail")
    return {"ok": ok, "n_passed": n_passed, "n_tests": len(inputs), "error": error}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    json.dump(grade_payload(payload), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
