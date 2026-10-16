# SPDX-FileCopyrightText: 2026 Khalid Tounoussi <khalid.tounoussi.1@ens.etsmtl.ca>
#
# SPDX-License-Identifier: MIT

"""Consensus clarification.

Asks one atomic question exactly as GatedClarification does, then draws several
independent implementations from the same solve prompt and keeps the first one unless
the others agree, by execution, on different behaviour.

Everything up to and including the first implementation is GatedClarification's code
path, unchanged: same prompts, same repair loop, same order of calls. The first
implementation is therefore distributed exactly like GatedClarification's single output,
and it is what this algorithm returns unless the evidence below says otherwise.

Selection uses no model judgement. The extra implementations run in the SDK sandbox on
inputs the model proposes from the task (inputs only -- never expected outputs), and on
examples the task itself states with their results. Two implementations agree when every
informative input gives outputs that compare equal under the evaluator's own generic
rule: Python equality, and its tolerance for floats. Shared crashes never count as
agreement. The first implementation is replaced only when a strictly larger group of
implementations agree with each other than agree with it; on a tie, or when no input
tells the implementations apart, the first one is kept.

Team: STIL-ETS
Team Members: Khalid Tounoussi, Issam Oukhay
Main Contact: khalid.tounoussi.1@ens.etsmtl.ca
"""

from __future__ import annotations

import ast
import base64
import json
import re
import zlib
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

GATE_TEMPLATE = """
You are reviewing a coding task before implementing it.

Task (function to implement: `{entry_point}`):
{prompt}

Decide whether the task is underspecified: is there a fact you would have to guess,
where guessing wrong would make a correct-looking implementation fail the user's tests?

Ignore stylistic choices, naming, and anything you can reasonably infer from the
description or examples. Only a genuine behavioural blocker counts.

If everything needed is present, reply with exactly:
NO_QUESTION

Otherwise reply with exactly one line:
QUESTION: <a single, specific question about exactly one missing fact>

The question must target one fact only, admit one unambiguous answer, and must not
ask for the algorithm, the implementation, or the test cases.
""".strip()

SOLVE_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves
the following problem:

{prompt}
{clarification}
Enclose your solution in ```python and ```.
""".strip()

CLARIFICATION_BLOCK = """
The user was asked a clarifying question and answered:
Q: {question}
A: {answer}

Follow that answer exactly, even where it contradicts your first reading of the task.
"""

EMPTY_SOLUTION = """```python
def {entry_point}(*args, **kwargs):
    raise NotImplementedError
```"""

REPAIR_TEMPLATE = """
Your response did not contain a valid fenced Python solution.
Return only the complete implementation of `{entry_point}`, enclosed in ```python and ```.
""".strip()


def usable_solution(response: str, entry_point: str) -> bool:
    """True when the response carries a fenced block that is real, runnable code.

    A fence alone is not enough: an empty block and a syntax error both parse out of
    the fence happily, and either one would silently pass for a solution.
    """
    try:
        code = _validate_and_parse_evalplus_result(response)
    except ValueError:
        return False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in ast.walk(tree)
    )


def complete_code(
    env: ClarificationEnvironment,
    prompt: str,
    entry_point: str,
    repair_attempts: int,
) -> str:
    """Ask for an implementation, nudging the model until the fence parses.

    Always returns a string, so an exhausted budget can never hand the evaluator None.
    """
    messages = [{"role": "user", "content": prompt}]
    last = ""

    for attempt in range(repair_attempts + 1):
        try:
            response = env.llm(messages)
        except LimitsExceededException:
            break

        last = response
        if usable_solution(response, entry_point):
            return response

        if attempt == repair_attempts:
            break

        messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": REPAIR_TEMPLATE.format(entry_point=entry_point)},
        ]

    return last if last else EMPTY_SOLUTION.format(entry_point=entry_point)


def parse_question(verdict: str, ask_when_unsure: bool) -> str | None:
    """Extract the single question from a gate verdict, or None.

    Model prose is never forwarded to the user verbatim: an unrecognisable verdict
    falls back to ``ask_when_unsure`` rather than sending whatever was produced.
    """
    lines = [line.strip() for line in verdict.splitlines() if line.strip()]

    if any(line.upper().strip("`*. ") == "NO_QUESTION" for line in lines):
        return None

    questions = [
        line.split(":", 1)[1].strip()
        for line in lines
        if line.upper().lstrip("`*- ").startswith("QUESTION:")
    ]

    if len(questions) == 1 and questions[0]:
        return questions[0]

    if ask_when_unsure and len(lines) == 1 and lines[0].endswith("?"):
        return lines[0]

    return None


PROBE_TEMPLATE = """
You are preparing inputs to exercise implementations of `{entry_point}`.

Task:
{prompt}
{clarification}
Write up to {count} different calls a user's test suite would plausibly make: the inputs of
the examples in the task, typical inputs, and the boundary cases the task implies (empty or
single-element inputs, ties, duplicates, zero or negative numbers, as applicable). Every
input must be valid for the task as described.

Reply with a single Python list literal and nothing else. Each element is a tuple of the
positional arguments of one call, e.g. [(3, [1, 2]), (0, [])]. A call with one argument is
a one-element tuple, e.g. [("abc",), ("",)]. Use only literals: numbers, strings, booleans,
None, lists, tuples and dicts. Do not include expected outputs or any code.
""".strip()

EXAMPLES_TEMPLATE = """
Copy the worked examples that the task below states explicitly, where both a call and its
expected result are written out. Copy only what is written; never invent, complete or
correct an example.
{override}
Task:
{prompt}

Reply with a single Python list literal and nothing else. Each element is a list of three
items: the exact text of the example as it appears in the task, a tuple of the call's
positional arguments, and the expected result. A call with one argument uses a one-element
tuple. Use only literals. Reply [] if the task states no such example.
""".strip()

EXAMPLES_OVERRIDE = """
The user answered a clarifying question. Leave out any example that this answer changes or
contradicts:
Q: {question}
A: {answer}
"""

# Runs inside env.exec_code's container, never in this process. Placeholders are filled
# with repr() of plain literals. The generic comparison mirrors the evaluator's own
# assertion: `out == exp`, and for floats atol 1e-6 / rtol 1e-7. Candidates run as
# __main__, as the evaluator runs them.
HARNESS = r"""
import ast
import contextlib
import copy
import io
import json
import signal
import sys
import time

CODES = __CODES__
PROBE_SOURCES = __PROBES__
EXAMPLE_SOURCES = __EXAMPLES__
ENTRY = __ENTRY__
STARTED = time.monotonic()
DEADLINE = STARTED + 15.0


class HarnessTimeout(BaseException):
    pass


def on_alarm(signum, frame):
    raise HarnessTimeout()


signal.signal(signal.SIGALRM, on_alarm)
sys.stdin = io.StringIO("")


def bounded(fn, seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def quiet(fn):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn()


def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return len(x) > 0 and all(isinstance(i, float) for i in x)
    return False


def same(a, b):
    try:
        if a == b:
            return True
    except Exception:
        pass
    try:
        if is_floats(a) and is_floats(b):
            xs = [a] if isinstance(a, float) else list(a)
            ys = [b] if isinstance(b, float) else list(b)
            return len(xs) == len(ys) and all(
                abs(x - y) <= 1e-6 + 1e-7 * abs(y) for x, y in zip(xs, ys)
            )
    except Exception:
        pass
    return False


def load(code):
    namespace = {"__name__": "__main__"}
    bounded(lambda: quiet(lambda: exec(compile(code, "<candidate>", "exec"), namespace)), 2.0)
    fn = namespace.get(ENTRY)
    if not callable(fn):
        raise LookupError(ENTRY)
    return fn


def call(fn, args):
    if fn is None:
        return ("absent", None)
    try:
        return ("ok", bounded(lambda: quiet(lambda: fn(*copy.deepcopy(args))), 1.0))
    except HarnessTimeout:
        return ("timeout", None)
    except BaseException as error:
        return ("error", type(error).__name__)


def literals(sources):
    out = []
    for source in sources:
        try:
            out.append(ast.literal_eval(source))
        except Exception:
            pass
    return out


fns, status = [], []
for code in CODES:
    if code is None:
        fns.append(None)
        status.append("unusable")
        continue
    if time.monotonic() > DEADLINE:
        fns.append(None)
        status.append("skipped")
        continue
    try:
        fns.append(load(code))
        status.append("loaded")
    except HarnessTimeout:
        fns.append(None)
        status.append("load-timeout")
    except BaseException as error:
        fns.append(None)
        status.append("load-error:" + type(error).__name__)

examples = [e for e in literals(EXAMPLE_SOURCES) if isinstance(e, tuple) and len(e) == 2]
probes = [p for p in literals(PROBE_SOURCES) if isinstance(p, tuple)]

active = []
for args, expected in examples:
    if time.monotonic() > DEADLINE:
        break
    passes = []
    for fn in fns:
        kind, value = call(fn, args)
        passes.append(kind == "ok" and same(value, expected))
    if any(passes):
        active.append(passes)

rows = []
for args in probes:
    if time.monotonic() > DEADLINE:
        break
    rows.append([call(fn, args) for fn in fns])

loaded = sum(fn is not None for fn in fns)
informative = []
for row in rows:
    ok = sum(kind == "ok" for kind, _ in row)
    if ok >= 2 and 2 * ok >= loaded:
        informative.append(row)


def agree(i, j):
    for row in informative:
        (ka, va), (kb, vb) = row[i], row[j]
        if ka != "ok" or kb != "ok" or not same(va, vb):
            return False
    return True


support = []
for i, fn in enumerate(fns):
    if fn is None:
        support.append(0)
    elif not informative:
        support.append(1)
    else:
        support.append(1 + sum(1 for j, g in enumerate(fns) if j != i and g is not None and agree(i, j)))

print("CANDIDATE-EXEC:" + json.dumps({
    "status": status,
    "support": support,
    "n_probes": len(probes),
    "n_run": len(rows),
    "n_informative": len(informative),
    "n_examples": len(examples),
    "n_active": len(active),
    "ex_pass": [[col[i] for col in active] for i in range(len(fns))],
    "seconds": round(time.monotonic() - STARTED, 3),
}))
"""

SUMMARY_MARK = "CANDIDATE-EXEC:"
AUDIT_MARK = "# CANDIDATE-AUDIT "
RULES = ("first", "consensus", "plurality", "examples", "examples_then_consensus")


def fenced_code(response: str) -> str | None:
    """The exact code the evaluator would run for this response, or None."""
    try:
        return _validate_and_parse_evalplus_result(response)
    except ValueError:
        return None


def literal_list(reply: str) -> list | None:
    """First Python list literal in a reply, parsed without executing anything."""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply or "", flags=re.DOTALL)
    body = fence.group(1) if fence else (reply or "")
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start or end - start > 20000:
        return None
    try:
        value = ast.literal_eval(body[start : end + 1])
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, list) else None


def portable(value: Any) -> str | None:
    """repr() of a literal that parses back to an equal value, else None."""
    try:
        source = repr(value)
        if len(source) > 2000 or ast.literal_eval(source) != value:
            return None
        return source
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def parse_probes(reply: str, limit: int) -> list[str]:
    out = []
    for item in literal_list(reply) or []:
        source = portable(item if isinstance(item, tuple) else (item,))
        if source and source not in out:
            out.append(source)
        if len(out) >= limit:
            break
    return out


def written_in(expected: Any, span: str) -> bool:
    """True when the expected value visibly appears in the quoted example text."""
    forms = [repr(expected), str(expected)]
    try:
        forms.append(json.dumps(expected))
    except (TypeError, ValueError):
        pass
    target = compact(span)
    return any(compact(form) and compact(form) in target for form in forms)


def parse_examples(reply: str, prompt: str, limit: int = 6) -> list[str]:
    """Examples whose quoted text occurs in the task and visibly contains the result.

    The extractor may only transcribe: an example is kept when its quoted text is found in
    the task and the expected value it reports is written inside that text.
    """
    task = compact(prompt)
    out = []
    for item in literal_list(reply) or []:
        if not (isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[0], str)):
            continue
        span, args, expected = item
        if not compact(span) or compact(span) not in task or not written_in(expected, span):
            continue
        source = portable((args if isinstance(args, tuple) else (args,), expected))
        if source and source not in out:
            out.append(source)
        if len(out) >= limit:
            break
    return out


def read_summary(output: str, k: int) -> dict[str, Any] | None:
    for line in reversed((output or "").splitlines()):
        if line.startswith(SUMMARY_MARK):
            try:
                data = json.loads(line[len(SUMMARY_MARK) :])
            except ValueError:
                return None
            ok = (
                isinstance(data, dict)
                and isinstance(data.get("support"), list)
                and len(data["support"]) == k
                and isinstance(data.get("ex_pass"), list)
                and len(data["ex_pass"]) == k
            )
            return data if ok else None
    return None


def select(rule: str, summary: dict[str, Any] | None, min_support: int) -> int:
    """Index of the implementation to return. Every doubt resolves to 0.

    first        -- always implementation 0 (what GatedClarification returns).
    consensus    -- replace 0 only by a group of at least `min_support` agreeing
                    implementations that is larger than 0's own group.
    plurality    -- replace 0 by any strictly larger group (comparison rule).
    examples     -- replace 0 only if it fails an example the task states that some
                    implementation satisfies, by the first one satisfying all of them.
    examples_then_consensus -- the example rule first; otherwise consensus restricted to
                    implementations that satisfy every applicable example.
    """
    if rule == "first" or not summary:
        return 0
    support = summary["support"]
    loaded = [i for i, s in enumerate(support) if s > 0]
    if not loaded:
        return 0
    informative = summary.get("n_informative", 0) > 0

    def group_pick(pool: list[int], needed: int) -> int:
        if not pool or not informative:
            return 0
        top = max(support[i] for i in pool)
        if support[0] >= top or top < needed:
            return 0
        return min(i for i in pool if support[i] == top)

    if rule == "consensus":
        return group_pick(loaded, min_support)
    if rule == "plurality":
        return group_pick(loaded, 1)

    active = summary.get("n_active", 0)
    ex_pass = summary["ex_pass"]
    eligible = [i for i in loaded if all(ex_pass[i])] if active else list(loaded)

    if rule == "examples":
        if not active or 0 in eligible or not eligible:
            return 0
        return min(eligible)
    if rule == "examples_then_consensus":
        if active and 0 not in eligible:
            if not eligible:
                return 0
            top = max(support[i] for i in eligible)
            return min(i for i in eligible if support[i] == top)
        return group_pick(eligible, min_support)
    return 0


def audit_line(payload: dict[str, Any]) -> str:
    blob = zlib.compress(json.dumps(payload).encode("utf-8"))
    return AUDIT_MARK + base64.b64encode(blob).decode("ascii")


class ConsensusClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # --- identical to GatedClarification ---
        # Retries allowed when a reply does not contain a usable ```python block.
        "repair_attempts": 2,
        # Treat an unrecognisable gate verdict as a question when it reads like one.
        "ask_when_unsure": True,
        # --- selection among independent implementations ---
        # Implementations drawn from the unchanged solve prompt; the first is the default.
        "samples": 5,
        # Inputs the model proposes for execution; inputs only, never expected outputs.
        "max_probes": 8,
        # Group size the stricter `consensus` rule requires; unused by `plurality`.
        "min_support": 3,
        # One of RULES; see `select`.
        "selection": "plurality",
        # Transcribe examples the task states, for the example-based rules and audit.
        "use_examples": True,
        # Development only: append an inert comment recording every implementation and the
        # execution summary, so each implementation can be scored offline.
        "audit": False,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]
        clarification = ""
        question = answer = None

        # GatedClarification, unchanged ------------------------------------------------
        if env.can_ask():
            try:
                verdict = env.llm(
                    GATE_TEMPLATE.format(prompt=problem["prompt"], entry_point=entry_point)
                )
            except LimitsExceededException:
                verdict = ""

            question = parse_question(verdict, self.config["ask_when_unsure"])
            if question:
                try:
                    answer = env.ask_human(question)
                    clarification = CLARIFICATION_BLOCK.format(question=question, answer=answer)
                except TooManyQuestionException:
                    pass

        solve_prompt = SOLVE_TEMPLATE.format(
            prompt=problem["prompt"],
            entry_point=entry_point,
            clarification=clarification,
        )
        responses = [complete_code(env, solve_prompt, entry_point, self.config["repair_attempts"])]
        # responses[0] is exactly what GatedClarification would return. -----------------

        summary, status = None, "ok"
        try:
            summary, status = self.gather(
                env, problem, entry_point, solve_prompt, responses, clarification, question, answer
            )
        except Exception as error:  # evidence gathering may never cost the default answer
            status = "error:" + repr(error)[:120]

        index = select(self.config["selection"], summary, self.config["min_support"])
        index = index if 0 <= index < len(responses) else 0
        return self.finish(responses, index, summary, status)

    def gather(
        self, env, problem, entry_point, solve_prompt, responses, clarification, question, answer
    ):
        config = self.config
        for _ in range(config["samples"] - 1):
            if env.prompt_cost >= 0.5 * env.prompt_budget:
                break
            responses.append(
                complete_code(env, solve_prompt, entry_point, config["repair_attempts"])
            )

        codes = [fenced_code(r) if usable_solution(r, entry_point) else None for r in responses]
        if sum(code is not None for code in codes) < 2:
            return None, "too-few-usable"

        probes = []
        if env.prompt_cost < 0.5 * env.prompt_budget:
            try:
                probes = parse_probes(
                    env.llm(
                        PROBE_TEMPLATE.format(
                            entry_point=entry_point,
                            prompt=problem["prompt"],
                            clarification=clarification,
                            count=config["max_probes"],
                        )
                    ),
                    config["max_probes"],
                )
            except LimitsExceededException:
                probes = []

        examples = []
        if config["use_examples"] and env.prompt_cost < 0.5 * env.prompt_budget:
            override = EXAMPLES_OVERRIDE.format(question=question, answer=answer) if answer else ""
            try:
                examples = parse_examples(
                    env.llm(EXAMPLES_TEMPLATE.format(override=override, prompt=problem["prompt"])),
                    problem["prompt"],
                )
            except LimitsExceededException:
                examples = []

        if not probes and not examples:
            return None, "no-evidence"

        script = (
            HARNESS.replace("__CODES__", repr(codes))
            .replace("__PROBES__", repr(probes))
            .replace("__EXAMPLES__", repr(examples))
            .replace("__ENTRY__", repr(entry_point))
        )
        summary = read_summary(env.exec_code(script), len(responses))
        return summary, ("ok" if summary else "no-summary")

    def finish(self, responses, index, summary, status):
        if not self.config["audit"]:
            return responses[index]
        codes = [fenced_code(r) for r in responses]
        line = audit_line(
            {
                "v": 1,
                "rule": self.config["selection"],
                "min_support": self.config["min_support"],
                "selected": index,
                "status": status,
                "codes": codes,
                "summary": summary,
            }
        )
        if codes[index] is not None:
            return f"```python\n{codes[index]}\n{line}\n```"
        return f"{responses[index]}\n{line}"
