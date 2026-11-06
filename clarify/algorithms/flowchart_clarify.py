# SPDX-FileCopyrightText: 2026 Mohamed Issam Daoud Ben Messaoud <issammohameddaoud@gmail.com>
#
# SPDX-License-Identifier: MIT

"""FlowChartClarify.

Models each coding requirement as an algorithm flowchart, treats every node whose
behaviour the requirement does not pin down as a candidate fork, prunes forks that
the prompt's own doctests already resolve, verifies that the two interpretations
really diverge by executing them on an in-domain witness input, and detects
description-vs-example contradictions by running every text-only reading on the
examples. The highest-ranked verified fork becomes the single clarification question.

Team: Hougar
Team Members: Mohammed Issam Daoud Ben Messaoud
Main Contact: issammohameddaoud@gmail.com
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

# Pruning reasons that only mean "below our confidence bar"; such forks are still
# reasonable questions when we always ask. Doctest-resolved, equivalent or
# out-of-domain forks are never asked.
SOFT_PRUNE_PREFIXES = ("PRUNED_BELOW_TURN_", "PRUNED_BY_CONVENTION_OR_BLAST")

TURN_THRESHOLDS = {
    1: {"min_ndcg": 3.4, "max_convention_conf": 0.80, "min_domain_validity": 0.80},
    2: {"min_ndcg": 4.9, "max_convention_conf": 0.55, "min_domain_validity": 0.88},
}

FLOWCHART_ANALYST_PROMPT = """
You are a senior software engineer reviewing a coding requirement before implementing it.
First model the intended algorithm as a small FLOWCHART, then decide whether any node of that flowchart is NOT determined by the requirement.

### Target Function Name
{entry_point}

### Problem Specification
{prompt}

### Extracted Inline Doctest Examples from Prompt
{extracted_doctests_summary}

### Clarifications Already Received
{clarification_history}

### Step-by-Step Analysis
1. Input Precondition Domain (D_in):
   Infer the constraints on valid inputs implied by the nouns, qualifiers, type hints and doctest examples. Anything already fixed by the natural meaning of the domain terms is part of D_in and MUST NOT be questioned.
2. Algorithm Flowchart:
   Draw the algorithm the requirement describes as a flowchart with at most 12 nodes, listed in execution order.
   Node types: "start", "process" (a computation step), "decision" (a branch on a condition), "end" (return the result).
   Model what the requirement asks for, not low-level implementation details. Label edges leaving a decision with the outcome (e.g. "yes"/"no").
   Each node label must commit to exactly ONE behaviour. Never hedge inside a label ("X or Y", "raise error / return None", "(or ...)").
   If you cannot commit to one behaviour at a node without guessing, that node is a candidate fork: report it in step 4 (the filters below decide whether it is worth asking).
   Do not silently bake an assumption into a node (e.g. "remove duplicates") when the requirement does not state it; that is also a candidate fork.
3. Doctest Gate (critical, prevents over-asking):
   For each node, check whether the text, the doctests and the clarifications already received determine exactly one behaviour. If one interpretation would give the WRONG output on any provided doctest (and the text does not explicitly contradict that doctest), the node is ALREADY DETERMINED. Do NOT report it.
4. Candidate Forks (only nodes NOT determined, strictly within D_in):
   - CONTRADICTION: the text explicitly states X, but a doctest or assertion requires Y.
   - LEXICAL_OR_DOMAIN_FORK: a process step's verb or noun admits two different algorithms that BOTH satisfy all doctests.
   - ORDER_OR_LOGIC_FORK: steps can be ordered or combined in two valid ways that BOTH satisfy all doctests but give different results on another valid input.
   - MISSING_BOUNDARY_GUARD: a decision the flowchart needs (e.g. empty input, ties, duplicates) is reachable by valid inputs, not covered by doctests, and has no standard Python convention.
   - STATE_MUTATION_FORK: unclear whether the input is modified in place or a new value is returned, and neither the signature nor the doctests resolve it.
   Do not report forks that common Python conventions settle. If every node is determined, return an empty "candidate_forks" list; that is the expected answer for well-specified requirements.
5. Executable Branches:
   A branch that raises an exception is allowed (e.g. for boundary forks): write it so the lambda actually raises.
   For each fork, give single-line Python lambdas (`lambda_expr` for Branch A and Branch B) taking exactly the same parameters as `{entry_point}`, and an in-domain `witness_args` list on which the two lambdas return DIFFERENT values.

6. Text-Only Readings:
   Give `text_readings`: a list of single-line Python lambdas (same parameters as `{entry_point}`), one for EACH plausible way to read the natural-language description ALONE, IGNORING all examples and doctests.
   If the description has only one sensible reading, give exactly one lambda. If a word is ambiguous (e.g. "invert" = reverse or negate), give one lambda per reading (at most 3).
   These are used to detect description-vs-example contradictions automatically: a contradiction exists only if NO reading matches the examples.

Return ONLY a valid JSON object matching this exact schema:
{
  "input_domain_preconditions": "concise summary of D_in",
  "text_readings": ["lambda lst: lst[::-1]", "lambda lst: [-v for v in lst]"],
  "flowchart": {
    "nodes": [
      {"id": "S", "type": "start", "label": "Receive lst"},
      {"id": "N1", "type": "process", "label": "Invert lst"},
      {"id": "E", "type": "end", "label": "Return result"}
    ],
    "edges": [
      {"from": "S", "to": "N1", "label": ""},
      {"from": "N1", "to": "E", "label": ""}
    ]
  },
  "candidate_forks": [
    {
      "fork_id": "F1",
      "node_id": "N1",
      "defect_type": "LEXICAL_OR_DOMAIN_FORK",
      "guard_condition": "concise description of what is undetermined at this node",
      "domain_validity": 0.95,
      "convention_confidence": 0.35,
      "blast_radius": "HIGH",
      "branch_a": {"label": "reverse the element order", "lambda_expr": "lambda lst: lst[::-1]"},
      "branch_b": {"label": "negate each element", "lambda_expr": "lambda lst: [-v for v in lst]"},
      "witness_args": [[1, 2, 3]],
      "expected_output_a": "[3, 2, 1]",
      "expected_output_b": "[-1, -2, -3]",
      "clarification_question": "Single concise question contrasting Branch A and Branch B on the witness example."
    }
  ]
}
""".strip()

FALLBACK_QUESTION_PROMPT = """
You are about to implement the function `{entry_point}` for the requirement below, and you may ask the user exactly ONE clarifying question first.

### Problem Specification
{prompt}

### Your Flowchart of the Algorithm
{flowchart}

Ask the single question whose answer most changes what the function returns. The question must:
- target ONE specific requirement (atomic), about observable behaviour, not implementation details;
- not be answerable by reading the specification or its examples;
- not ask for test cases or hidden tests;
- admit one unambiguous answer (offer two concrete alternatives on a small example input when possible).

Return ONLY the question, on one line.
""".strip()

FINAL_CODE_SYNTHESIS_PROMPT = """
Provide a clean Python implementation solving the following problem.

### Target Function Name
{entry_point}

### Problem Specification
{prompt}

### Clarifications & Resolved Decisions
{clarification_history}

### Strict Output Rules
1. Implement ONLY the required function(s) and necessary standard library imports.
2. Do NOT include any main block, example usage, print statements, or extra commentary.
3. Enclose your code in a ```python ... ``` code block.
4. Where a clarification answer conflicts with the specification or its examples, follow the clarification answer.
""".strip()


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extracts a JSON object from an LLM response safely."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    return {"flowchart": {"nodes": [], "edges": []}, "candidate_forks": []}


def _normalize_branch(raw: Any, default_label: str) -> dict[str, str]:
    """Accepts {"label", "lambda_expr"} dicts, bare lambda strings or bare labels."""
    if isinstance(raw, dict):
        return {
            "label": str(raw.get("label") or default_label),
            "lambda_expr": str(raw.get("lambda_expr") or ""),
        }
    text = str(raw or "").strip()
    if text.startswith("lambda"):
        return {"label": default_label, "lambda_expr": text}
    return {"label": text or default_label, "lambda_expr": ""}


def _normalize_analysis(data: Any) -> dict[str, Any]:
    """Coerces the LLM's JSON into the expected shape so malformed fields never crash a task."""
    if not isinstance(data, dict):
        data = {}
    flow = data.get("flowchart")
    if not isinstance(flow, dict):
        flow = {}
    nodes = [n for n in flow.get("nodes") or [] if isinstance(n, dict) and n.get("id") is not None]
    edges = [e for e in flow.get("edges") or [] if isinstance(e, dict)]

    forks = []
    raw_forks = data.get("candidate_forks")
    for fork in raw_forks if isinstance(raw_forks, list) else []:
        if not isinstance(fork, dict):
            continue
        fork = dict(fork)
        fork["branch_a"] = _normalize_branch(fork.get("branch_a"), "Branch A")
        fork["branch_b"] = _normalize_branch(fork.get("branch_b"), "Branch B")
        witness = fork.get("witness_args")
        fork["witness_args"] = witness if isinstance(witness, list) else [witness]
        forks.append(fork)

    readings = data.get("text_readings") or data.get("text_reading_lambda") or []
    if isinstance(readings, str):
        readings = [readings]
    if not isinstance(readings, list):
        readings = []

    normalized = dict(data)
    normalized["flowchart"] = {"nodes": nodes, "edges": edges}
    normalized["candidate_forks"] = forks
    normalized["text_readings"] = [str(r).strip() for r in readings if str(r).strip()]
    return normalized


def _extract_doctests_from_prompt(
    prompt_text: str, entry_point: str
) -> list[tuple[list[Any], Any]]:
    """
    Parses inline doctest examples (`>>> entry_point(...)` -> expected_value)
    and simple `assert entry_point(...) == expected_value` from the prompt docstring.
    Returns a list of (args_list, expected_value) tuples.
    """
    if not entry_point or entry_point == "solution":
        return []

    extracted: list[tuple[list[Any], Any]] = []
    lines = prompt_text.splitlines()
    n_lines = len(lines)

    i = 0
    while i < n_lines:
        line = lines[i].strip()
        # Pattern 1: Doctest `>>> func(...)`
        if (
            line.startswith(">>>")
            and entry_point in line
            and _extract_inline_example(line, entry_point) is None
        ):
            call_Dynamic = line[3:].strip()
            # Handle multi-line doctest calls if needed
            while call_Dynamic.endswith(",") or call_Dynamic.count("(") > call_Dynamic.count(")"):
                if i + 1 < n_lines and lines[i + 1].strip().startswith("..."):
                    i += 1
                    call_Dynamic += " " + lines[i].strip()[3:].strip()
                else:
                    break
            # Next non-empty line is expected output
            if i + 1 < n_lines:
                exp_line = lines[i + 1].strip()
                if exp_line and not exp_line.startswith(">>>") and not exp_line.startswith("#"):
                    parsed_pair = _parse_call_and_expected(call_Dynamic, exp_line, entry_point)
                    if parsed_pair is not None:
                        extracted.append(parsed_pair)
                    i += 1
        # Pattern 2: `assert f(...) == y`, and prose examples such as
        # `f(...) ==> y`, `f(...) ➞ y`, `f(...) -> y`, `f(...) should return y`
        elif entry_point + "(" in line.replace("`", ""):
            parsed_pair = _extract_inline_example(line, entry_point)
            if parsed_pair is not None:
                extracted.append(parsed_pair)
        i += 1

    # De-duplicate while preserving order (at most 8 examples are used)
    unique: list[tuple[list[Any], Any]] = []
    for pair in extracted:
        if pair not in unique:
            unique.append(pair)
    return unique[:8]


_EXAMPLE_ARROW_RE = re.compile(
    r"^\s*(?:==>|=>|➞|→|->|==|=|should\s+return|returns?|gives|yields|outputs?|is)\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _extract_inline_example(line: str, entry_point: str) -> tuple[list[Any], Any] | None:
    """Finds `entry_point(...)` in a line and parses the value after an arrow/`==`."""
    line = line.replace("`", " ")
    needle = entry_point + "("
    idx = line.find(needle)
    while idx > 0 and (line[idx - 1].isalnum() or line[idx - 1] in "_."):
        idx = line.find(needle, idx + 1)
    if idx == -1:
        return None

    depth, end, in_str = 0, None, None
    for j in range(idx + len(entry_point), len(line)):
        ch = line[j]
        if in_str:
            if ch == in_str and line[j - 1] != "\\":
                in_str = None
            continue
        if ch in "'\"":
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        return None

    match = _EXAMPLE_ARROW_RE.match(line[end + 1 :])
    if not match:
        return None
    expected_str = match.group(1).rstrip(".,;")
    return _parse_call_and_expected(line[idx : end + 1], expected_str, entry_point)


def _parse_call_and_expected(
    call_str: str, expected_str: str, entry_point: str
) -> tuple[list[Any], Any] | None:
    """Safely parses `entry_point(arg1, arg2)` and `expected_str` using `ast.literal_eval`."""
    try:
        tree = ast.parse(call_str, mode="eval")
        if not isinstance(tree.body, ast.Call):
            return None
        func_node = tree.body.func
        if not isinstance(func_node, ast.Name) or func_node.id != entry_point:
            return None
        if tree.body.keywords:
            return None
        args_list = [ast.literal_eval(arg) for arg in tree.body.args]
        expected_val = ast.literal_eval(expected_str)
        return args_list, expected_val
    except Exception:
        return None


RAISED_TAG = "__raised__"


def _exc_name(exc: BaseException) -> str:
    """Exception class name, e.g. "ValueError" (without touching dunder attributes)."""
    return repr(exc).split("(", 1)[0]


def _outcome_repr(value: Any) -> str:
    """Human-readable outcome; a raised exception is shown as `raises ValueError`."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == RAISED_TAG:
        return f"raises {value[1]}"
    return repr(value)


BLOCKED_TOKENS = ("input(", "open(", "os.", "sys.", "subprocess", "eval(", "exec(", "import")

# Script run inside the SDK sandbox (env.exec_code). It evaluates every queued
# lambda on its arguments with a per-call timeout and prints the outcomes as JSON.
SANDBOX_SCRIPT = """
import json, signal, time
import math, collections, itertools, re, heapq, functools, string
from typing import *

JOBS = __JOBS__
TIMEOUT = __TIMEOUT__
BUDGET = 20.0


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _on_alarm)
started = time.monotonic()
results = {}
for key, src, args, allow_raise in JOBS:
    if time.monotonic() - started > BUDGET:
        results[key] = ["error", "Budget"]
        continue
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        fn = eval(src)
        try:
            results[key] = ["ok", repr(fn(*args))]
        except _Timeout:
            raise
        except Exception as exc:
            name = type(exc).__name__
            results[key] = ["raised", name] if allow_raise else ["error", name]
    except _Timeout:
        results[key] = ["error", "Timeout"]
    except Exception as exc:
        results[key] = ["error", type(exc).__name__]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
print("@@FLOWCHART_RESULTS@@" + json.dumps(results) + "@@END@@")
"""


def _job_key(lambda_str: str, args: list[Any], allow_raise: bool) -> str:
    return hashlib.sha1(f"{lambda_str}\x00{args!r}\x00{allow_raise}".encode()).hexdigest()


def _as_args(args: Any) -> list[Any]:
    return args if isinstance(args, list) else [args]


def _decode_value(text: str) -> Any:
    """Turns the sandbox's repr() back into a value; unparsable reprs stay comparable as text."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return ("__repr__", text)


class _LambdaRunner:
    """Executes all lambdas of one analysis turn in a single env.exec_code call."""

    def __init__(self, timeout: float = 0.5) -> None:
        self.timeout = timeout
        self.jobs: dict[str, tuple[str, list[Any], bool]] = {}
        self.results: dict[str, tuple[bool, Any]] = {}

    def add(self, lambda_str: str, args: Any, allow_raise: bool = False) -> None:
        src = str(lambda_str or "").strip()
        if not src.startswith("lambda") or any(tok in src for tok in BLOCKED_TOKENS):
            return
        self.jobs[_job_key(src, _as_args(args), allow_raise)] = (src, _as_args(args), allow_raise)

    def execute(self, env: ClarificationEnvironment) -> None:
        pending = [(k, *job) for k, job in self.jobs.items() if k not in self.results]
        if not pending:
            return
        lines = []
        for key, src, args, allow_raise in pending:
            lines.append(f"    ({key!r}, {src!r}, {args!r}, {allow_raise!r}),")
        jobs_literal = "[\n" + "\n".join(lines) + "\n]"
        script = SANDBOX_SCRIPT.replace("__JOBS__", jobs_literal).replace(
            "__TIMEOUT__", repr(float(self.timeout))
        )
        try:
            output = str(env.exec_code(script))
        except Exception:
            output = ""
        match = re.search(r"@@FLOWCHART_RESULTS@@(.*?)@@END@@", output, re.S)
        decoded: dict[str, Any] = {}
        if match:
            try:
                decoded = json.loads(match.group(1))
            except json.JSONDecodeError:
                decoded = {}
        for key, *_ in pending:
            status, payload = decoded.get(key, ["error", "ExecFailed"])
            if status == "ok":
                self.results[key] = (True, _decode_value(payload))
            elif status == "raised":
                self.results[key] = (True, (RAISED_TAG, payload))
            else:
                self.results[key] = (False, payload)

    def __call__(self, lambda_str: str, args: Any, allow_raise: bool = False) -> tuple[bool, Any]:
        src = str(lambda_str or "").strip()
        if not src.startswith("lambda"):
            return False, "NotALambda"
        if any(tok in src for tok in BLOCKED_TOKENS):
            return False, "BlockedBuiltin"
        return self.results.get(_job_key(src, _as_args(args), allow_raise), (False, "NotRun"))


def _values_equal(val1: Any, val2: Any) -> bool:
    """Checks equality with float tolerance."""
    if type(val1) is not type(val2) and not (
        isinstance(val1, (int, float)) and isinstance(val2, (int, float))
    ):
        return False
    if isinstance(val1, float) or isinstance(val2, float):
        try:
            return abs(float(val1) - float(val2)) < 1e-6
        except Exception:
            return False
    return val1 == val2


def _check_doctest_consistency(
    lambda_a: str,
    lambda_b: str,
    doctests: list[tuple[list[Any], Any]],
    defect_type: str,
    runner: _LambdaRunner,
) -> tuple[str, int, int]:
    """
    Runs both `lambda_a` and `lambda_b` against extracted prompt doctests.
    Returns (status, pass_count_a, pass_count_b) where status is one of:
    - "NO_DOCTESTS": No doctests were extracted or lambdas couldn't be evaluated on them.
    - "DISAMBIGUATED_BY_DOCTEST": Exactly one branch passes all doctests while the other fails at least one!
    - "BOTH_CONSISTENT_WITH_DOCTESTS": Both branches pass all doctests (true underspecification!).
    - "CONTRADICTS_DOCTEST": Doctest reveals conflict/contradiction.
    """
    if not doctests or not lambda_a or not lambda_b:
        return "NO_DOCTESTS", 0, 0

    evaluated_count = 0
    pass_a = 0
    pass_b = 0

    for args_list, expected_val in doctests:
        ok_a, res_a = runner(lambda_a, args_list)
        ok_b, res_b = runner(lambda_b, args_list)
        if not ok_a or not ok_b:
            continue
        evaluated_count += 1
        if _values_equal(res_a, expected_val):
            pass_a += 1
        if _values_equal(res_b, expected_val):
            pass_b += 1

    if evaluated_count == 0:
        return "NO_DOCTESTS", 0, 0

    if pass_a == evaluated_count and pass_b == evaluated_count:
        return "BOTH_CONSISTENT_WITH_DOCTESTS", pass_a, pass_b

    if (pass_a == evaluated_count) != (pass_b == evaluated_count):
        # One branch passes 100% of prompt doctests and the other fails!
        # Unless the LLM flagged an explicit text-vs-example CONTRADICTION,
        # the prompt's own doctests already disambiguate this fork!
        if defect_type == "CONTRADICTION":
            return "CONTRADICTS_DOCTEST", pass_a, pass_b
        return "DISAMBIGUATED_BY_DOCTEST", pass_a, pass_b

    # Neither branch matches the doctests. For a declared CONTRADICTION that can be
    # genuine; otherwise the LLM's lambdas are most likely just wrong, and this must
    # not be rewarded as evidence of ambiguity.
    if defect_type == "CONTRADICTION":
        return "CONTRADICTS_DOCTEST", pass_a, pass_b
    return "LAMBDAS_FAIL_DOCTESTS", pass_a, pass_b


def _trace_name(problem: dict[str, Any]) -> str:
    """Unique per variant: the SDK passes no task_id, and variants share the entry point."""
    entry_pt = str(problem.get("entry_point") or "solution")
    digest = hashlib.sha1(str(problem.get("prompt", "")).encode("utf-8")).hexdigest()[:8]
    return re.sub(r"\W", "_", f"{entry_pt}_{digest}")


def _flowchart_node_order(analysis: dict[str, Any]) -> dict[str, int]:
    """Maps each non-start flowchart node id to its 1-based position in execution order."""
    flow = analysis.get("flowchart") or {}
    order: dict[str, int] = {}
    position = 0
    for node in flow.get("nodes", []):
        if not isinstance(node, dict) or node.get("id") is None:
            continue
        if str(node.get("type", "")).lower() == "start":
            continue
        position += 1
        order[str(node["id"])] = position
    return order


AUTO_CONTRADICTION_GUARD = "description contradicts a provided example"


def _detect_text_example_conflict(
    text_readings: list[str],
    doctests: list[tuple[list[Any], Any]],
    entry_point: str,
    anchor_node: str,
    runner: _LambdaRunner,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Deterministic contradiction check (no extra LLM call).

    Runs every plausible text-only reading of the requirement on the prompt's own
    examples. If at least one reading matches all examples, the examples merely
    disambiguate the text: no contradiction. Only when every reading that runs
    cleanly disagrees with some example do the description and examples conflict.
    """
    if not text_readings or not doctests:
        return None

    first_mismatch = None
    clean_readings = 0
    for reading in text_readings[:3]:
        mismatch = None
        clean = True
        for args_list, expected_val in doctests:
            ok, got = runner(reading, args_list)
            if not ok:
                clean = False  # unreliable lambda: ignore this reading entirely
                break
            if mismatch is None and not _values_equal(got, expected_val):
                mismatch = (reading, args_list, expected_val, got)
        if not clean:
            continue
        clean_readings += 1
        if mismatch is None:
            return None  # this reading agrees with every example
        if first_mismatch is None:
            first_mismatch = mismatch

    if clean_readings == 0 or first_mismatch is None:
        return None

    text_lambda, args_list, expected_val, got = first_mismatch
    call = f"{entry_point}({', '.join(repr(a) for a in args_list)})"
    fork = {
        "fork_id": "AUTO_CONTRADICTION",
        "node_id": anchor_node,
        "defect_type": "CONTRADICTION",
        "guard_condition": AUTO_CONTRADICTION_GUARD,
        "domain_validity": 1.0,
        "convention_confidence": 0.0,
        "blast_radius": "HIGH",
        "branch_a": {"label": "follow the description", "lambda_expr": text_lambda},
        "branch_b": {"label": "follow the example", "lambda_expr": ""},
        "witness_args": args_list,
        "clarification_question": (
            f"The description implies {call} returns {got!r}, but the example says it "
            f"returns {expected_val!r}. Which behaviour is correct?"
        ),
    }
    meta = {
        "reason": "VERIFIED_TEXT_EXAMPLE_CONFLICT",
        "doctest_status": "CONTRADICTS_DOCTEST",
        "verified_by_exec": True,
        "witness_args": args_list,
        "output_a": repr(got),
        "output_b": repr(expected_val),
        "ndcg_score": 10.0,
    }
    return fork, meta


def _verify_and_score_fork(
    fork: dict[str, Any],
    turn_idx: int,
    doctests: list[tuple[list[Any], Any]],
    runner: _LambdaRunner,
) -> tuple[bool, float, dict[str, Any]]:
    """Verifies domain validity, doctest consistency, witness divergence, and turn threshold."""
    thresholds = TURN_THRESHOLDS.get(turn_idx, TURN_THRESHOLDS[2])

    blast = str(fork.get("blast_radius", "HIGH")).upper()
    defect_type = str(fork.get("defect_type", "ORDER_OR_LOGIC_FORK")).upper()

    try:
        convention_conf = float(fork.get("convention_confidence", 0.5))
    except (TypeError, ValueError):
        convention_conf = 0.5

    try:
        domain_validity = float(fork.get("domain_validity", 0.9))
    except (TypeError, ValueError):
        domain_validity = 0.9

    # Filter 1: Precondition Domain Subtyping Check
    if domain_validity < thresholds["min_domain_validity"]:
        return (
            False,
            0.0,
            {"reason": "PRUNED_OUT_OF_DOMAIN_WITNESS", "domain_validity": domain_validity},
        )

    # Filter 2: Turn-aware Blast Radius & Convention Prior Check
    if blast == "LOW" or convention_conf > thresholds["max_convention_conf"]:
        return (
            False,
            0.0,
            {"reason": "PRUNED_BY_CONVENTION_OR_BLAST", "convention_conf": convention_conf},
        )

    if turn_idx >= 2 and blast != "HIGH":
        return False, 0.0, {"reason": "PRUNED_SECONDARY_TURN_NON_HIGH_BLAST"}

    lambda_a = str(fork.get("branch_a", {}).get("lambda_expr", "")).strip()
    lambda_b = str(fork.get("branch_b", {}).get("lambda_expr", "")).strip()
    witness_args = fork.get("witness_args", [])

    # Filter 3: Automated Doctest Contract Check (Eliminates Over-Asking on examples!)
    dt_status, pass_a, pass_b = _check_doctest_consistency(
        lambda_a, lambda_b, doctests, defect_type, runner
    )
    if dt_status == "DISAMBIGUATED_BY_DOCTEST":
        return (
            False,
            0.0,
            {
                "reason": "PRUNED_DISAMBIGUATED_BY_DOCTEST",
                "pass_a": pass_a,
                "pass_b": pass_b,
            },
        )

    verified_by_exec = False
    out_a_repr = str(fork.get("expected_output_a", "")).strip()
    out_b_repr = str(fork.get("expected_output_b", "")).strip()

    # Filter 4: In-Memory Differential Witness Execution
    if lambda_a and lambda_b and isinstance(witness_args, list):
        allow_raise = defect_type in ("MISSING_BOUNDARY_GUARD", "CONTRADICTION")
        ok_a, val_a = runner(lambda_a, witness_args, allow_raise=allow_raise)
        ok_b, val_b = runner(lambda_b, witness_args, allow_raise=allow_raise)
        if ok_a and ok_b:
            if _values_equal(val_a, val_b):
                return (
                    False,
                    0.0,
                    {"reason": "PRUNED_OBSERVATIONALLY_EQUIVALENT", "output": _outcome_repr(val_a)},
                )
            verified_by_exec = True
            out_a_repr, out_b_repr = _outcome_repr(val_a), _outcome_repr(val_b)

    if not verified_by_exec:
        if not out_a_repr or not out_b_repr:
            return False, 0.0, {"reason": "PRUNED_UNVERIFIABLE_NO_OUTPUTS"}
        if out_a_repr == out_b_repr:
            return False, 0.0, {"reason": "PRUNED_IDENTICAL_EXPECTED_OUTPUT"}

    defect_weights = {
        "CONTRADICTION": 3.6,
        "LEXICAL_OR_DOMAIN_FORK": 2.8,
        "ORDER_OR_LOGIC_FORK": 2.4,
        "STATE_MUTATION_FORK": 1.8,
        "MISSING_BOUNDARY_GUARD": 1.5,
    }
    base_weight = defect_weights.get(defect_type, 2.0)

    try:
        step_order = max(1, int(fork.get("step_order", 1)))
    except (TypeError, ValueError):
        step_order = 1

    doctest_bonus = (
        0.45 if dt_status in ("BOTH_CONSISTENT_WITH_DOCTESTS", "CONTRADICTS_DOCTEST") else 0.0
    )

    ndcg_score = round(
        base_weight * (1.0 + 1.0 / step_order) * domain_validity
        + (0.3 if verified_by_exec else 0.0)
        + (0.5 if blast == "HIGH" else 0.0)
        + doctest_bonus,
        3,
    )

    # Filter 5: Dynamic Turn-Discounted Score Gate
    if ndcg_score < thresholds["min_ndcg"]:
        return (
            False,
            ndcg_score,
            {
                "reason": f"PRUNED_BELOW_TURN_{turn_idx}_THRESHOLD",
                "ndcg_score": ndcg_score,
                "required": thresholds["min_ndcg"],
            },
        )

    return (
        True,
        ndcg_score,
        {
            "reason": "VERIFIED_DIVERGENT_FORK",
            "doctest_status": dt_status,
            "verified_by_exec": verified_by_exec,
            "witness_args": witness_args,
            "output_a": out_a_repr,
            "output_b": out_b_repr,
            "ndcg_score": ndcg_score,
        },
    )


def _strip_main_and_prints(code_str: str) -> str:
    """Removes main execution blocks and top-level example prints using AST."""
    dunder_name = "_" + "_name_" + "_"
    dunder_main = "_" + "_main_" + "_"
    try:
        tree = ast.parse(code_str)
        kept_nodes = []
        for node in tree.body:
            if isinstance(node, ast.If):
                test_src = ast.unparse(node.test)
                if dunder_name in test_src and dunder_main in test_src:
                    continue
            kept_nodes.append(node)
        tree.body = kept_nodes
        return ast.unparse(tree).strip()
    except Exception:
        lines = []
        for line in code_str.splitlines():
            if dunder_name in line and dunder_main in line:
                break
            lines.append(line)
        return "\n".join(lines).strip()


def _mermaid_id(raw: Any) -> str:
    """Mermaid-safe node id (prefixed so ids like `end` cannot clash with keywords)."""
    return "n_" + re.sub(r"\W", "_", str(raw))


def _mermaid_text(raw: Any, limit: int = 90) -> str:
    """Mermaid-safe label text."""
    text = re.sub(r"[|\n\r]", " ", str(raw)).strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def compile_trace_to_mermaid(
    entry_point: str, analysis: dict[str, Any], evaluations: list[dict[str, Any]]
) -> str:
    """Renders the algorithm flowchart plus every candidate fork as a Mermaid flowchart.

    Verified forks are highlighted in amber with both branches and their witness
    outputs; pruned forks are shown greyed out with the pruning reason.
    """
    flow = analysis.get("flowchart") or {}
    nodes = [n for n in flow.get("nodes", []) if isinstance(n, dict) and n.get("id") is not None]
    edges = [e for e in flow.get("edges", []) if isinstance(e, dict)]

    lines = [f"%% Flowchart trace for {entry_point}", "flowchart TD"]
    shapes = {"start": ('(["', '"])'), "end": ('(["', '"])'), "decision": ('{"', '"}')}
    known_ids = set()
    for node in nodes:
        nid = _mermaid_id(node["id"])
        known_ids.add(str(node["id"]))
        left, right = shapes.get(str(node.get("type", "process")).lower(), ('["', '"]'))
        lines.append(f"  {nid}{left}{_mermaid_text(node.get('label', node['id']))}{right}")
    if not nodes:
        lines.append(f'  n_S(["{_mermaid_text(entry_point)}"])')
        known_ids.add("S")

    for edge in edges:
        label = _mermaid_text(edge.get("label", ""), 30)
        arrow = f" -->|{label}| " if label else " --> "
        lines.append(f"  {_mermaid_id(edge.get('from'))}{arrow}{_mermaid_id(edge.get('to'))}")

    default_anchor = str(nodes[0]["id"]) if nodes else "S"
    for idx, item in enumerate(evaluations, 1):
        fork, meta, kept = item["fork"], item["meta"], item["kept"]
        anchor_raw = fork.get("node_id")
        anchor = _mermaid_id(anchor_raw if str(anchor_raw) in known_ids else default_anchor)
        fid = f"fork{idx}"
        label_a = _mermaid_text(fork.get("branch_a", {}).get("label", "Branch A"), 60)
        label_b = _mermaid_text(fork.get("branch_b", {}).get("label", "Branch B"), 60)
        if kept:
            out_a = _mermaid_text(meta.get("output_a"), 40)
            out_b = _mermaid_text(meta.get("output_b"), 40)
            lines.append(f'  {fid}_A["A: {label_a} → {out_a}"]')
            lines.append(f'  {fid}_B["B: {label_b} → {out_b}"]')
            lines.append(f"  {anchor} -.->|ASK score={meta.get('ndcg_score')}| {fid}_A")
            lines.append(f"  {anchor} -.-> {fid}_B")
            lines.append(f"  style {anchor} fill:#FEF3C7,stroke:#D97706,stroke-width:2px")
            lines.append(f"  style {fid}_A fill:#FDE68A,stroke:#D97706")
            lines.append(f"  style {fid}_B fill:#FDE68A,stroke:#D97706")
        else:
            reason = _mermaid_text(meta.get("reason", "pruned"), 50)
            lines.append(f'  {fid}_P["pruned: {label_a} vs {label_b} ({reason})"]')
            lines.append(f"  {anchor} -.- {fid}_P")
            lines.append(f"  style {fid}_P fill:#F3F4F6,stroke:#9CA3AF,color:#6B7280")
    return "\n".join(lines)


class FlowChartClarify(ClarificationAlgorithmBase):
    """FlowChart-based clarification: ask the best verified flowchart fork."""

    DEFAULT_CONFIG = {
        # Always ask one question on the first turn (TDS charges only ~4% per turn).
        "always_ask_first_turn": True,
        # Upper bound on analysis turns; the environment's own turn limit still applies.
        "max_clarification_turns": 2,
        # Per-lambda timeout (seconds) inside the env.exec_code sandbox.
        "lambda_timeout": 0.5,
        # Optional folder for JSON/Mermaid traces (None = no traces written).
        "trace_dir": None,
    }

    def trace_dir(self) -> str | None:
        """Folder for traces (config `trace_dir`, e.g. `--trace_dir=traces`)."""
        return self.config.get("trace_dir") or None

    def format_history(self, qa_pairs: list[tuple[str, str]]) -> str:
        if not qa_pairs:
            return "None."
        return "\n\n".join(
            f"Question {i}: {q}\nUser Answer {i}: {a}" for i, (q, a) in enumerate(qa_pairs, 1)
        )

    def format_doctests_summary(
        self, doctests: list[tuple[list[Any], Any]], entry_point: str
    ) -> str:
        if not doctests:
            return "No inline doctest examples detected."
        items = []
        for args_list, expected_val in doctests[:6]:
            args_str = ", ".join(repr(a) for a in args_list)
            items.append(f"- {entry_point}({args_str}) == {repr(expected_val)}")
        return "\n".join(items)

    def maybe_save_trace(
        self,
        problem: dict[str, Any],
        turn_idx: int,
        analysis: dict[str, Any],
        evaluations: list[dict[str, Any]],
    ) -> None:
        if not self.trace_dir():
            return
        try:
            out_dir = Path(self.trace_dir())
            out_dir.mkdir(parents=True, exist_ok=True)
            entry_pt = str(problem.get("entry_point") or "solution")
            task_id = _trace_name(problem)
            mermaid_str = compile_trace_to_mermaid(entry_pt, analysis, evaluations)

            trace_file = out_dir / f"{task_id}_turn{turn_idx}.json"
            mmd_file = out_dir / f"{task_id}_turn{turn_idx}.mmd"

            trace_file.write_text(
                json.dumps({"analysis": analysis, "evaluations": evaluations}, indent=2),
                encoding="utf-8",
            )
            mmd_file.write_text(mermaid_str, encoding="utf-8")
        except Exception:
            pass

    def maybe_save_qa(
        self,
        problem: dict[str, Any],
        qa_pairs: list[tuple[str, str]],
        final_code: str | None = None,
    ) -> None:
        """Writes the questions actually asked, the answers, and the final code."""
        if not self.trace_dir():
            return
        try:
            out_dir = Path(self.trace_dir())
            out_dir.mkdir(parents=True, exist_ok=True)
            entry_pt = str(problem.get("entry_point") or "solution")
            task_id = _trace_name(problem)
            payload = {
                "task_id": problem.get("task_id", entry_pt),
                "num_questions": len(qa_pairs),
                "qa": [
                    {"turn": i, "question": q, "answer": a} for i, (q, a) in enumerate(qa_pairs, 1)
                ],
                "final_code": final_code,
            }
            (out_dir / f"{task_id}_qa.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def synthesize_final_code(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        qa_pairs: list[tuple[str, str]],
    ) -> str:
        entry_point = str(problem.get("entry_point") or "solution")
        messages = [
            {
                "role": "user",
                "content": (
                    FINAL_CODE_SYNTHESIS_PROMPT.replace("{prompt}", problem["prompt"])
                    .replace("{entry_point}", entry_point)
                    .replace("{clarification_history}", self.format_history(qa_pairs))
                ),
            }
        ]
        response = env.llm(messages)
        try:
            parsed = _validate_and_parse_evalplus_result(response)
            return _strip_main_and_prints(parsed)
        except ValueError:
            fence_instruction = "Enclose ONLY the function implementation inside ```python and ``` without any main block."
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": fence_instruction},
            ]
            retry_resp = env.llm(messages)
            try:
                parsed = _validate_and_parse_evalplus_result(retry_resp)
                return _strip_main_and_prints(parsed)
            except ValueError:
                return _strip_main_and_prints(retry_resp)

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        qa_pairs: list[tuple[str, str]] = []
        try:
            return self.run_pipeline(env, problem, qa_pairs)
        except TooManyQuestionException:
            return self.finish(env, problem, qa_pairs)
        except Exception:
            # Safety net: never lose a task to an analysis bug; still produce code,
            # using whatever clarifications were already obtained.
            return self.finish(env, problem, qa_pairs)

    def run_pipeline(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        qa_pairs: list[tuple[str, str]],
    ) -> str:
        entry_point = str(problem.get("entry_point") or "solution")
        prompt_text = problem["prompt"]
        doctests = _extract_doctests_from_prompt(prompt_text, entry_point)
        doctests_summary = self.format_doctests_summary(doctests, entry_point)

        asked_signatures: set[str] = set()

        max_turns = int(self.config.get("max_clarification_turns", 2))
        for turn in range(1, max_turns + 1):
            if turn > 1 and not self.can_still_ask(env):
                break
            architect_messages = [
                {
                    "role": "user",
                    "content": (
                        FLOWCHART_ANALYST_PROMPT.replace("{prompt}", prompt_text)
                        .replace("{entry_point}", entry_point)
                        .replace("{extracted_doctests_summary}", doctests_summary)
                        .replace("{clarification_history}", self.format_history(qa_pairs))
                    ),
                }
            ]
            raw_analysis = env.llm(architect_messages)
            analysis = _normalize_analysis(_extract_json_payload(raw_analysis))
            candidate_forks = analysis.get("candidate_forks", [])
            node_order = _flowchart_node_order(analysis)
            runner = self.prepare_runner(env, analysis, doctests)

            verified_forks: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            all_evaluations: list[dict[str, Any]] = []

            for fork in candidate_forks:
                if not isinstance(fork, dict):
                    continue
                sig = f"{fork.get('defect_type')}:{fork.get('guard_condition')}"
                if sig in asked_signatures:
                    continue
                # Earlier flowchart nodes rank higher: the step order comes from the
                # node's position in the flowchart rather than a number the LLM picks.
                fork["step_order"] = node_order.get(
                    str(fork.get("node_id")), fork.get("step_order", 1)
                )
                kept, score, meta = _verify_and_score_fork(
                    fork, turn_idx=turn, doctests=doctests, runner=runner
                )
                all_evaluations.append({"fork": fork, "kept": kept, "meta": meta})
                if kept:
                    verified_forks.append((score, fork, meta))

            # Deterministic description-vs-example conflict check (asked once at most).
            auto_sig = f"CONTRADICTION:{AUTO_CONTRADICTION_GUARD}"
            if auto_sig not in asked_signatures:
                flow_nodes = (analysis.get("flowchart") or {}).get("nodes", [])
                anchor = next(
                    (
                        str(n.get("id"))
                        for n in flow_nodes
                        if isinstance(n, dict) and str(n.get("type", "")).lower() != "start"
                    ),
                    "S",
                )
                readings = (
                    analysis.get("text_readings") or analysis.get("text_reading_lambda") or []
                )
                if isinstance(readings, str):
                    readings = [readings]
                readings = [str(r).strip() for r in readings if str(r).strip()]
                conflict = _detect_text_example_conflict(
                    readings, doctests, entry_point, anchor, runner
                )
                if conflict is not None:
                    auto_fork, auto_meta = conflict
                    all_evaluations.append({"fork": auto_fork, "kept": True, "meta": auto_meta})
                    verified_forks.append((auto_meta["ndcg_score"], auto_fork, auto_meta))

            self.maybe_save_trace(problem, turn, analysis, all_evaluations)

            if not verified_forks:
                if not (turn == 1 and self.config.get("always_ask_first_turn", True)):
                    return self.finish(env, problem, qa_pairs)
                # Always-ask mode: best softly-pruned fork, else one LLM-written question.
                soft = [
                    e
                    for e in all_evaluations
                    if not e["kept"]
                    and str(e["meta"].get("reason", "")).startswith(SOFT_PRUNE_PREFIXES)
                    and str(e["fork"].get("clarification_question", "")).strip()
                ]
                soft.sort(key=lambda e: float(e["meta"].get("ndcg_score") or 0.0), reverse=True)
                if soft:
                    question = str(soft[0]["fork"]["clarification_question"]).strip()
                else:
                    question = self.fallback_question(env, problem, analysis)
                if not question:
                    return self.finish(env, problem, qa_pairs)
                try:
                    qa_pairs.append((question, env.ask_human(question)))
                except TooManyQuestionException:
                    break
                continue

            verified_forks.sort(key=lambda item: item[0], reverse=True)
            _best_score, top_fork, top_meta = verified_forks[0]

            sig = f"{top_fork.get('defect_type')}:{top_fork.get('guard_condition')}"
            asked_signatures.add(sig)

            question = str(top_fork.get("clarification_question", "")).strip()
            if not question:
                b_a = top_fork.get("branch_a", {}).get("label", "Option A")
                b_b = top_fork.get("branch_b", {}).get("label", "Option B")
                question = (
                    f"Regarding {top_fork.get('guard_condition', 'this requirement')}: "
                    f"should the function {b_a} (returning `{top_meta.get('output_a')}`) "
                    f"or {b_b} (returning `{top_meta.get('output_b')}`)?"
                )

            try:
                answer = env.ask_human(question)
                qa_pairs.append((question, answer))
            except TooManyQuestionException:
                break

        return self.finish(env, problem, qa_pairs)

    def can_still_ask(self, env: ClarificationEnvironment) -> bool:
        """Skips a further (costly) analysis turn when the environment allows no more questions."""
        try:
            return bool(env.can_ask())
        except Exception:
            return True

    def prepare_runner(
        self,
        env: ClarificationEnvironment,
        analysis: dict[str, Any],
        doctests: list[tuple[list[Any], Any]],
    ) -> _LambdaRunner:
        """Queues every lambda execution this turn needs and runs them in ONE env.exec_code call."""
        runner = _LambdaRunner(timeout=float(self.config.get("lambda_timeout", 0.5)))
        for fork in analysis.get("candidate_forks", []):
            lambda_a = fork["branch_a"]["lambda_expr"]
            lambda_b = fork["branch_b"]["lambda_expr"]
            for args_list, _expected in doctests:
                runner.add(lambda_a, args_list)
                runner.add(lambda_b, args_list)
            defect_type = str(fork.get("defect_type", "ORDER_OR_LOGIC_FORK")).upper()
            allow_raise = defect_type in ("MISSING_BOUNDARY_GUARD", "CONTRADICTION")
            runner.add(lambda_a, fork["witness_args"], allow_raise)
            runner.add(lambda_b, fork["witness_args"], allow_raise)
        for reading in analysis.get("text_readings", [])[:3]:
            for args_list, _expected in doctests:
                runner.add(reading, args_list)
        runner.execute(env)
        return runner

    def fallback_question(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        analysis: dict[str, Any],
    ) -> str:
        entry_point = str(problem.get("entry_point") or "solution")
        content = (
            FALLBACK_QUESTION_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry_point)
            .replace("{flowchart}", json.dumps(analysis.get("flowchart") or {}, indent=1))
        )
        try:
            response = str(env.llm([{"role": "user", "content": content}])).strip()
        except Exception:
            return ""
        lines = [ln.strip().strip("`").strip() for ln in response.splitlines() if ln.strip()]
        return lines[0] if lines else ""

    def finish(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        qa_pairs: list[tuple[str, str]],
    ) -> str:
        code = self.synthesize_final_code(env, problem, qa_pairs)
        self.maybe_save_qa(problem, qa_pairs, code)
        return code
