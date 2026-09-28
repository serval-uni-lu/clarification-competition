# SPDX-FileCopyrightText: 2026 Md Zahidul Haque <mhaque@wm.edu>
#
# SPDX-License-Identifier: MIT

"""InfoGain Clarifier.

Samples several candidate implementations, executes them on shared probe
inputs to find where they behave differently, and asks the single atomic
question whose answer is expected to remove the most behavioural uncertainty
(expected information gain over behaviour clusters). Remaining unasked
decisions are resolved with a default-interpretation prior; the final program
is selected by tests derived from the user's answer.

Team: RankOne
Team Members: Md Zahidul Haque
Main Contact: mhaque@wm.edu
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

CODE_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}
{context}
Only define the function (and helpers). Do not add a `__main__` block, prints, input() or asserts.
Enclose your solution in ```python and ```.
""".strip()

AUDIT_TEMPLATE = """
You review a coding task before implementing it. List the decisions an implementer must make that the
task leaves open: ambiguous wording, missing details, or statements that contradict each other or the
examples. Always consider the function signature (number, order and types of arguments), the return
type and exact output format, and edge-case behaviour.

### Task (implement `{entry_point}`)
{prompt}

For each open decision write ONE atomic question that asks for exactly one fact, has a single objective
answer, and does not ask for the implementation or test cases. Give 2-3 plausible answers, the answer a
careful engineer would assume by default, and a criticality from 1 (cosmetic) to 3 (changes outputs).

Return JSON only, inside ```json and ```:
[{{"id": "d1", "question": "...", "options": ["...", "..."], "default": "...", "criticality": 3}}]
Return [] if nothing is open.
""".strip()

PROBE_TEMPLATE = """
Task (implement `{entry_point}`):
{prompt}

Write {num_probes} diverse argument lists for calling `{entry_point}` that would expose differences
between plausible interpretations of the task (edge cases, boundary values, unusual formats).
Return ONE Python literal: a list where each element is a list of positional arguments.
Enclose it in ```python and ```. Use only literals (no function calls, no imports).
""".strip()

LABEL_TEMPLATE = """
Task:
{prompt}

Open decisions:
{decisions}

Candidate implementations:
{candidates}

For every candidate and every decision, state which option index (0-based) the candidate implements,
or -1 if it cannot be told. Return JSON only, inside ```json and ```:
{{"c0": {{"d1": 0, "d2": 1}}, "c1": {{...}}}}
""".strip()

ANSWER_TESTS_TEMPLATE = """
Task (implement `{entry_point}`):
{prompt}

The user clarified:
Q: {question}
A: {answer}

Write 3-5 Python assert statements that call `{entry_point}` and check ONLY the behaviour fixed by this
clarification. Do not define `{entry_point}`. Enclose them in ```python and ```.
""".strip()

RUNNER = r"""
import copy, io, itertools, json, signal, sys, time
from math import inf, nan

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
except Exception:
    pass


class _Timeout(BaseException):
    pass


def _timeout(signum, frame):
    raise _Timeout()


def _call(fn, seconds):
    # Re-arm on every call: a candidate may have replaced the handler or cancelled the alarm.
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn()
    finally:
        signal.alarm(0)
        sys.stdout = saved


_REAL_STDOUT = sys.stdout
_DEADLINE = time.monotonic() + {deadline}


def _emit(out):
    _REAL_STDOUT.write("__RESULT__" + json.dumps(out) + "\n")
    _REAL_STDOUT.flush()


def _canon(value):
    if isinstance(value, (set, frozenset)):
        return type(value).__name__ + "{{" + ", ".join(sorted(_canon(v) for v in value)) + "}}"
    if isinstance(value, dict):
        items = sorted(_canon(k) + ": " + _canon(v) for k, v in value.items())
        return "{{" + ", ".join(items) + "}}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_canon(v) for v in value)
        return ("[" + inner + "]") if isinstance(value, list) else ("(" + inner + ")")
    if hasattr(value, "__next__"):
        return "iter" + _canon(list(itertools.islice(value, 1000)))
    text = repr(value)
    return type(value).__name__ if " at 0x" in text else text


_SOURCES = {sources}
_PROBES = {probes}
_ENTRY = {entry!r}
_fns = {{}}
_out = {{}}
for _name, _src in _SOURCES.items():
    _ns = {{"__name__": "_candidate"}}
    try:
        _call(lambda: exec(_src, _ns), 2)
        _fns[_name] = _ns[_ENTRY]
        _out[_name] = []
    except BaseException as _e:
        _out[_name] = "LOAD:" + type(_e).__name__
_emit(_out)
for _args in _PROBES:
    _row = {{}}
    for _name, _fn in _fns.items():
        if time.monotonic() > _DEADLINE:
            break
        try:
            _row[_name] = _call(lambda: _canon(_fn(*copy.deepcopy(_args))), 2)
        except BaseException as _e:
            _row[_name] = "ERR:" + type(_e).__name__
    if len(_row) < len(_fns):
        break
    for _name, _value in _row.items():
        _out[_name].append(_value)
    _emit(_out)
"""

TEST_RUNNER = r"""
import copy, io, itertools, json, signal, sys, time
from math import inf, nan

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
except Exception:
    pass


class _Timeout(BaseException):
    pass


def _timeout(signum, frame):
    raise _Timeout()


def _call(fn, seconds):
    # Re-arm on every call: a candidate may have replaced the handler or cancelled the alarm.
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn()
    finally:
        signal.alarm(0)
        sys.stdout = saved


_REAL_STDOUT = sys.stdout
_DEADLINE = time.monotonic() + {deadline}


def _emit(out):
    _REAL_STDOUT.write("__RESULT__" + json.dumps(out) + "\n")
    _REAL_STDOUT.flush()


_SOURCES = {sources}
_TESTS = {tests}
_out = {{_name: 0 for _name in _SOURCES}}
_emit(_out)
for _t in _TESTS:
    _row = {{}}
    for _name, _src in _SOURCES.items():
        if time.monotonic() > _DEADLINE:
            break
        _ns = {{"__name__": "_candidate"}}
        try:
            _call(lambda: (exec(_src, _ns), exec(_t, _ns)), 3)
            _row[_name] = 1
        except BaseException:
            _row[_name] = 0
    if len(_row) < len(_SOURCES):
        break
    for _name, _passed in _row.items():
        _out[_name] += _passed
    _emit(_out)
"""

# The sandbox kills a run after 30 s; stop starting new rows well before that.
EXEC_DEADLINE_SECONDS = 15


def extract_block(text: str, lang: str) -> str | None:
    match = re.search(rf"```{lang}\s*\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1) if match else None


def sanitize(code: str, entry_point: str) -> str:
    """Keep definitions, imports and assignments; drop top-level side effects."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
    )
    tree.body = [node for node in tree.body if isinstance(node, keep)]
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    if entry_point not in defined:
        return code
    return ast.unparse(tree)


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def criticality(decision: dict) -> int:
    return as_int(decision.get("criticality"), 1)


def entropy(counts: list[float]) -> float:
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c > 0) if total else 0.0


class InfoGainClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "num_candidates": 4,  # candidate programs sampled before asking
        "num_probes": 8,  # probe inputs used to cluster candidate behaviour
        "num_final": 3,  # candidate programs sampled after the clarification
        "residual_prior": True,  # fix unasked decisions to the auditor's default interpretation
        "use_answer_tests": True,  # select the final program by tests derived from the answer
    }

    # LLM helpers -----------------------------------------------------------

    def ask_json(self, env, prompt):
        response = env.llm(prompt)
        block = extract_block(response, "json") or response
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None

    def generate(self, env, problem, context=""):
        prompt = (
            CODE_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{context}", context)
        )
        messages = [{"role": "user", "content": prompt}]
        for _ in range(2):
            response = env.llm(messages)
            try:
                return sanitize(
                    _validate_and_parse_evalplus_result(response), problem["entry_point"]
                )
            except ValueError as e:
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(e)},
                ]
        return None

    def audit(self, env, problem):
        decisions = self.ask_json(
            env,
            AUDIT_TEMPLATE.replace("{entry_point}", problem["entry_point"]).replace(
                "{prompt}", problem["prompt"]
            ),
        )
        if not isinstance(decisions, list):
            return []
        return [
            d
            for d in decisions
            if isinstance(d, dict) and d.get("question") and len(d.get("options", [])) >= 2
        ]

    def probes(self, env, problem):
        response = env.llm(
            PROBE_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{num_probes}", str(self.config["num_probes"]))
        )
        block = extract_block(response, "python")
        if not block:
            return None
        try:
            probes = ast.literal_eval(block.strip())
        except (ValueError, SyntaxError):
            return None
        if not isinstance(probes, list):
            return None
        return [p if isinstance(p, list | tuple) else [p] for p in probes]

    # Execution ---------------------------------------------------------------

    def execute(self, env, template, **fields):
        try:
            output = env.exec_code(template.format(**fields))
        except Exception:  # a sandbox failure degrades this step, not the whole task
            return None
        if "__RESULT__" not in output:
            return None
        try:
            return json.loads(output.rsplit("__RESULT__", 1)[1].strip().splitlines()[0])
        except (json.JSONDecodeError, IndexError):
            return None

    def behaviour_clusters(self, env, problem, candidates, probes):
        sources = {f"c{i}": c for i, c in enumerate(candidates)}
        results = self.execute(
            env,
            RUNNER,
            sources=repr(sources),
            probes=repr(probes),
            entry=problem["entry_point"],
            deadline=EXEC_DEADLINE_SECONDS,
        )
        if results is None:
            return {name: 0 for name in sources}
        signatures = {
            name: json.dumps(outs)
            for name, outs in results.items()
            if isinstance(outs, list) and outs
        }
        ids = {sig: i for i, sig in enumerate(dict.fromkeys(signatures.values()))}
        return {name: ids[sig] for name, sig in signatures.items()}

    # Question selection ------------------------------------------------------

    def information_gain(self, decision_id, labels, clusters, num_options):
        """Coverage-weighted H(cluster) - E_option[H(cluster | option)] over labelled candidates."""
        known = {}
        for name, cluster in clusters.items():
            row = labels.get(name)
            option = row.get(str(decision_id)) if isinstance(row, dict) else None
            if isinstance(option, str) and option.strip().isdigit():
                option = int(option)
            if type(option) is int and 0 <= option < num_options:
                known[name] = (option, cluster)
        if len(known) < 2:
            return 0.0
        total = Counter(cluster for _, cluster in known.values())
        by_option: dict[int, Counter] = {}
        for option, cluster in known.values():
            by_option.setdefault(option, Counter())[cluster] += 1
        n = len(known)
        conditional = sum(
            sum(c.values()) / n * entropy(list(c.values())) for c in by_option.values()
        )
        return n / len(clusters) * (entropy(list(total.values())) - conditional)

    def select_question(self, env, problem, decisions, candidates, clusters):
        if not decisions:
            return None
        ranked = sorted(decisions, key=lambda d: -criticality(d))
        if len(set(clusters.values())) < 2:
            return ranked[0]

        described = "\n".join(
            f"{d.get('id', f'd{i}')}: {d['question']} options={d['options']}"
            for i, d in enumerate(decisions)
        )
        shown = "\n\n".join(f"c{i}:\n```python\n{c}\n```" for i, c in enumerate(candidates))
        labels = self.ask_json(
            env,
            LABEL_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{decisions}", described)
            .replace("{candidates}", shown),
        )
        if not isinstance(labels, dict):
            return ranked[0]

        def score(item):
            i, d = item
            gain = self.information_gain(d.get("id", f"d{i}"), labels, clusters, len(d["options"]))
            return (gain, criticality(d))

        return max(enumerate(decisions), key=score)[1]

    # Final selection -----------------------------------------------------------

    def answer_tests(self, env, problem, question, answer):
        response = env.llm(
            ANSWER_TESTS_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{question}", question)
            .replace("{answer}", answer)
        )
        block = extract_block(response, "python") or ""
        try:
            tree = ast.parse(block)
        except SyntaxError:
            return [
                line.strip() for line in block.splitlines() if line.strip().startswith("assert")
            ]
        entry = problem["entry_point"]
        body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node.name == entry
            )
        ]
        asserts = [node for node in body if isinstance(node, ast.Assert)]
        if asserts:
            prelude = "\n".join(ast.unparse(n) for n in body if not isinstance(n, ast.Assert))
            return [f"{prelude}\n{ast.unparse(node)}".strip() for node in asserts]
        if any(isinstance(node, ast.Assert) for node in ast.walk(ast.Module(body, []))):
            return [ast.unparse(ast.Module(body, []))]
        return []

    def select_final(self, env, problem, finals, tests):
        if len(finals) == 1:
            return finals[0]
        scores = {}
        if tests:
            sources = {f"c{i}": c for i, c in enumerate(finals)}
            scores = (
                self.execute(
                    env,
                    TEST_RUNNER,
                    sources=repr(sources),
                    tests=repr(tests),
                    deadline=EXEC_DEADLINE_SECONDS,
                )
                or {}
            )
        best = max(scores.values(), default=0)
        pool = [c for i, c in enumerate(finals) if scores.get(f"c{i}", 0) == best] or finals
        # Tie-break: most common program text after normalisation, else the first.
        return Counter(pool).most_common(1)[0][0]

    # Main ----------------------------------------------------------------------

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        decisions = self.audit(env, problem)
        candidates = [
            c
            for c in (self.generate(env, problem) for _ in range(self.config["num_candidates"]))
            if c
        ]
        if not candidates:
            return ""

        probes = self.probes(env, problem)
        clusters = (
            self.behaviour_clusters(env, problem, candidates, probes)
            if probes
            else {f"c{i}": 0 for i in range(len(candidates))}
        )

        chosen = self.select_question(env, problem, decisions, candidates, clusters)
        question = chosen["question"] if chosen else None
        answer = None
        if question and env.can_ask():
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                answer = None

        context_lines = []
        if question and answer:
            context_lines.append(f"The user clarified: Q: {question} A: {answer}")
        if self.config["residual_prior"]:
            for d in decisions:
                if d is not chosen and d.get("default"):
                    context_lines.append(f"Assume: {d['question']} -> {d['default']}")
        context = ("\n" + "\n".join(context_lines) + "\n") if context_lines else ""

        finals = [
            c
            for c in (self.generate(env, problem, context) for _ in range(self.config["num_final"]))
            if c
        ] or candidates[:1]

        tests = []
        if self.config["use_answer_tests"] and question and answer:
            tests = self.answer_tests(env, problem, question, answer)
        return self.select_final(env, problem, finals, tests)
