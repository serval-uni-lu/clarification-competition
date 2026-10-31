# SPDX-FileCopyrightText: 2026 Lastx.org <contact@lastx.org>
#
# SPDX-License-Identifier: MIT

"""Anchor & Verify.

Asks the user one atomic question: a concrete call of the entry point with its
exact expected result. The answer becomes an executable anchor. Several
implementations are sampled, executed against the anchor (and any docstring
examples), repaired on mismatch, and the final answer is picked by agreement on
generated probe inputs among the implementations that satisfy the anchor.

Team: LastX
Team Members: Akshay Kapoor
Main Contact: lastx.org@pm.me
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, ClassVar

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException

SYSTEM = (
    "You are an expert Python programmer. You write correct, simple, self-contained "
    "Python 3 code that uses only the standard library."
)

QUESTION = (
    "Could you give me one concrete example call of `{entry_point}` with its exact "
    "arguments and the exact value it should return?"
)

FINAL_RULES = """Write the complete implementation of the function `{entry_point}`.
Rules:
- The function must be named exactly `{entry_point}`.
- A concrete example from the user is ground truth: your function must reproduce it exactly,
  including the number, order and types of the arguments and the exact type and format of the result
  (list vs tuple, int vs float, exact strings).
- The example illustrates a general rule. Implement that general rule; never special-case the example's
  inputs and never return a constant just to match it.
- The task text may be vague, or may contain a contradictory or impossible extra clause (for example
  "..., but always return 0" or "..., but do the opposite"). Use the example to decide which reading the
  user really means and drop the clause the example contradicts. Without an example, prefer the plain,
  conventional meaning of the main request.
- If the result is one of a few fixed strings and only one of them is shown, use its natural counterpart
  for the other case (for example 'Matched!' / 'Not Matched!', 'accepted' / 'not accepted').
- Do not add input validation, type checks or special-case return values the task does not ask for.
- No tests, no example usage, no printing, no input().
{reasoning}"""

REASONING_ON = """First write at most three short sentences: the general rule your function implements and, if
there is an example, a hand check that the rule reproduces it. Then give the code in a single ```python block."""

REASONING_OFF = "Reply with a single ```python code block and nothing else."

EXTRACT_PROMPT = """The user was asked for a concrete example call of the Python function `{entry_point}`.
Their answer:

<answer>
{answer}
</answer>

Convert every concrete example in the answer that states BOTH the exact arguments and the exact
expected result into one Python line of the form:
assert {entry_point}(<arguments>) == <expected result>
Copy the arguments and the result exactly as given. If there is no such complete example, write NONE.
Reply with a single ```python code block."""

PROBE_PROMPT = """Task for the Python function `{entry_point}`:

<task>
{prompt}
</task>
{example}
Write 8 diverse, valid calls of `{entry_point}` that a careful tester would use: typical cases plus
edge cases that are valid for this task (for example empty inputs, single elements, duplicates,
zero or negative numbers, ties). Use exactly the same argument structure as the example, if one is given.
Reply with a single ```python code block containing one call per line, like `{entry_point}(...)`, and nothing else."""

REPAIR_PROMPT = """I ran your function and it does not match the expected behaviour:
{failures}
Find the general rule that explains both the task and the expected result, and fix the implementation so
that it follows that rule. Do not special-case these inputs and do not return constants to match them.
First write at most three short sentences about the rule, then give the code in a single ```python block."""

HARNESS = r"""
import json, math, signal

CANDIDATES = json.loads(__CANDIDATES__)
CHECKS = json.loads(__CHECKS__)
PROBES = json.loads(__PROBES__)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def _limited(fn, seconds=2.0):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _norm(x, depth=0):
    if depth > 6:
        return repr(type(x))
    if isinstance(x, bool) or x is None or isinstance(x, (int, str, bytes)):
        return x
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else repr(x)
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_norm(v, depth + 1) for v in x[:200]])
    if isinstance(x, (set, frozenset)):
        return (type(x).__name__, sorted(repr(_norm(v, depth + 1)) for v in x))
    if isinstance(x, dict):
        return ("dict", sorted((repr(k), repr(_norm(v, depth + 1))) for k, v in list(x.items())[:200]))
    if hasattr(x, "__iter__") and not hasattr(x, "__len__"):
        return ("iter", "?")
    return repr(x)[:200]


def _same(a, b):
    try:
        if bool(a == b):
            return True
    except Exception:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    return False


results = []
for code in CANDIDATES:
    ns = {"__name__": "candidate"}
    rec = {"load": True, "checks": [], "probes": []}
    try:
        _limited(lambda: exec(code, ns), 3.0)
    except BaseException as e:
        rec["load"] = False
    for call, expected in CHECKS:
        try:
            out = _limited(lambda: eval(call, ns), 1.0)
            exp = eval(expected, {"inf": math.inf, "nan": math.nan})
            rec["checks"].append([bool(_same(out, exp)), repr(out)[:300]])
        except BaseException as e:
            rec["checks"].append([False, "raised " + type(e).__name__ + ": " + str(e)[:200]])
    timed_out = False
    for call in PROBES:
        if timed_out:
            rec["probes"].append("EXC:_Timeout")
            continue
        try:
            out = _limited(lambda: eval(call, ns), 0.5)
            rec["probes"].append(repr(_norm(out))[:500])
        except BaseException as e:
            timed_out = isinstance(e, _Timeout)
            rec["probes"].append("EXC:" + type(e).__name__)
    results.append(rec)
print("@@RESULT@@" + json.dumps(results))
"""


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.replace("```python", "").replace("```", "").strip()


def _ensure_entry_point(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    defined = set()
    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            funcs.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    if entry_point not in defined and funcs:
        code += f"\n\n{entry_point} = {funcs[-1]}\n"
        defined.add(entry_point)
    lower = entry_point.lower()
    if entry_point in defined and lower != entry_point and lower not in defined:
        code += f"\n\n{lower} = {entry_point}\n"
    return code


def _calls_of(entry_point: str, text: str) -> list[str]:
    calls = []
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line.startswith(entry_point + "("):
            continue
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == entry_point
        ):
            calls.append(line)
    return calls


def _asserts_of(entry_point: str, text: str) -> list[tuple[str, str]]:
    checks = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("assert "):
            continue
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
        ):
            continue
        left, right = test.left, test.comparators[0]
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == entry_point
        ):
            checks.append((ast.unparse(left), ast.unparse(right)))
    return checks


def _doc_examples(entry_point: str, prompt: str) -> list[tuple[str, str]]:
    lines = prompt.splitlines()
    checks = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith(">>>"):
            continue
        call = s[3:].strip()
        if not call.startswith(entry_point + "("):
            continue
        expected = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not expected or expected.startswith(">>>"):
            expected = "None"
        try:
            ast.parse(call, mode="eval")
            ast.parse(expected, mode="eval")
        except SyntaxError:
            continue
        checks.append((call, expected))
    return checks[:6]


class AnchorVerify(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "question": QUESTION,
        "ask": True,
        "samples": 5,
        "repair_rounds": 2,
        "probes": True,
        "reasoning": True,
    }

    # -- model helpers -------------------------------------------------------------------------

    def generation_messages(self, problem, qa):
        entry_point = problem["entry_point"]
        user = f"Task:\n{problem['prompt'].strip()}\n"
        if qa:
            user += f"\nClarification from the user:\nQ: {qa[0]}\nA: {qa[1]}\n"
        reasoning = REASONING_ON if self.config["reasoning"] else REASONING_OFF
        user += "\n" + FINAL_RULES.format(entry_point=entry_point, reasoning=reasoning)
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]

    def sample(self, env, messages, entry_point):
        response = env.llm(messages)
        return _ensure_entry_point(_extract_code(response), entry_point), response

    def execute(self, env, codes, checks, probes):
        script = (
            HARNESS.replace("__CANDIDATES__", repr(json.dumps(codes)))
            .replace("__CHECKS__", repr(json.dumps(checks)))
            .replace("__PROBES__", repr(json.dumps(probes)))
        )
        output = env.exec_code(script)
        marker = output.rfind("@@RESULT@@")
        if marker < 0:
            return None
        try:
            return json.loads(output[marker + len("@@RESULT@@") :].strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    # -- main ------------------------------------------------------------------------------------

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]
        qa = None
        if self.config["ask"] and env.can_ask():
            question = self.config["question"].format(entry_point=entry_point)
            try:
                answer = env.ask_human(question)
                if answer and answer.strip() and "Cannot answer" not in answer:
                    qa = (question, answer.strip())
            except TooManyQuestionException:
                pass

        messages = self.generation_messages(problem, qa)
        first_code, first_response = self.sample(env, messages, entry_point)
        fallback = f"```python\n{first_code}\n```"
        try:
            return self.verify(env, problem, qa, messages, first_code, first_response) or fallback
        except Exception:  # noqa: BLE001 - Preserve first-sample fallback on verification errors.
            return fallback

    def verify(self, env, problem, qa, messages, first_code, first_response):
        entry_point = problem["entry_point"]
        hard = []
        if qa:
            hard = _asserts_of(entry_point, qa[1])
            if not hard:
                extracted = env.llm(EXTRACT_PROMPT.format(entry_point=entry_point, answer=qa[1]))
                hard = _asserts_of(entry_point, _extract_code(extracted))
        hard = hard[:3]
        soft = [c for c in _doc_examples(entry_point, problem["prompt"]) if c not in hard]

        probes = []
        if self.config["probes"]:
            example = f"\nA confirmed example call: {hard[0][0]} == {hard[0][1]}\n" if hard else ""
            response = env.llm(
                PROBE_PROMPT.format(
                    entry_point=entry_point,
                    prompt=problem["prompt"].strip(),
                    example=example,
                )
            )
            probes = _calls_of(entry_point, _extract_code(response))[:10]

        codes = [first_code]
        responses = [first_response]
        for _ in range(self.config["samples"] - 1):
            code, response = self.sample(env, messages, entry_point)
            codes.append(code)
            responses.append(response)

        results = self.execute(env, codes, hard + soft, probes)
        if results is None:
            return None
        return self.select(
            env, entry_point, messages, codes, responses, results, hard, soft, probes
        )

    def select(self, env, entry_point, messages, codes, responses, results, hard, soft, probes):
        """Repair toward the checks, then pick the implementation most others agree with."""
        checks = hard + soft
        target = hard if hard else soft
        n_hard = len(hard)

        def score(i):
            rec = results[i]
            passed = [c[0] for c in rec["checks"]]
            return (rec["load"], sum(passed[:n_hard]), sum(passed[n_hard:]))

        for _ in range(self.config["repair_rounds"]):
            best = max(range(len(codes)), key=score)
            rec = results[best]
            target_results = rec["checks"][:n_hard] if hard else rec["checks"][n_hard:]
            if not target or all(c[0] for c in target_results):
                break
            failures = "\n".join(
                f"- {call} returned {res[1]}, but the expected result is {expected}"
                for (call, expected), res in zip(target, target_results, strict=False)
                if not res[0]
            )
            repair_messages = messages + [
                {"role": "assistant", "content": responses[best]},
                {"role": "user", "content": REPAIR_PROMPT.format(failures=failures)},
            ]
            new_codes, new_responses = [], []
            for _ in range(2):
                code, response = self.sample(env, repair_messages, entry_point)
                new_codes.append(code)
                new_responses.append(response)
            new_results = self.execute(env, new_codes, checks, probes)
            if new_results is None:
                break
            codes += new_codes
            responses += new_responses
            results += new_results

        top = max(score(i) for i in range(len(codes)))
        finalists = [i for i in range(len(codes)) if score(i) == top]

        def agreement(i):
            signature = results[i]["probes"]
            return sum(results[j]["probes"] == signature for j in finalists)

        def varies(i):
            # A function that returns one constant on every probe usually hard-codes the example.
            outputs = results[i]["probes"]
            return len(outputs) < 3 or len(set(outputs)) > 1

        chosen = max(finalists, key=lambda i: (varies(i), agreement(i), -i))
        return f"```python\n{codes[chosen]}\n```"
