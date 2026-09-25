# SPDX-FileCopyrightText: 2026 kelvin715 <yzhsk99@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Foresight Clarification.

Draft first, then ask about the draft's riskiest assumption. The assistant writes a
draft implementation, then spends its clarification turn on the single decision in
that draft most likely to make it return a different value than the user expects
(core meaning, exact result form, argument list, or a text/example conflict),
phrased as one atomic question. It regenerates the implementation from scratch with
the answer (authoritative where specific) and finalizes the code deterministically:
it keeps the right code block, strips top-level test code, binds the entry point,
restores helpers from the prompt, and tolerates arity mismatches on signature-less
tasks.

Team: SkyWalker
Team Members: kelvin715
Main Contact: yzhsk99@gmail.com
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SOLVE_TEMPLATE = """Implement the Python function `{entry_point}` for the task below.

<task>
{prompt}
</task>
{clarification}
Rules:
- Define a top-level function named exactly `{entry_point}`; the tests call it by that name.{signature_rule}
- Where the task is vague or silent, choose the most standard, conventional behaviour implied by the function name and the rest of the description.
- {output_rule}
"""

OUTPUT_RULES = {
    False: "Reply with one ```python code block containing the complete implementation and its imports only: no example usage, no tests, no prints, no input().",
    True: "First write a short <spec>...</spec> listing the precise requirements you will implement (arguments, exact form of the result, behaviour, special cases), applying the user's clarification if there is one and resolving any contradictions in the task. Then give one ```python code block containing the complete implementation and its imports only: no example usage, no tests, no prints, no input().",
}

CLARIFICATION_BLOCK = """
Before implementing, you asked the user clarifying questions and got these answers:
{pairs}
The user's answers are authoritative where they are specific: follow them even if they contradict parts of the task description (descriptions can contain mistakes). Everything the answers do not address still follows the task description.
"""

QUESTION_TEMPLATE = """A user gave you this coding task:

<task>
{prompt}
</task>
{previous}
The user's acceptance tests call `{entry_point}` on ordinary, valid inputs and compare the returned value exactly with what the user intended. The description may be vague, incomplete, or even contradict what the user really wants (an example, a type, a constraint or a claimed behaviour may be wrong).

Your current draft implementation:
```python
{candidate}
```

You may ask the user exactly ONE clarifying question before finalizing. Pick the uncertainty that most likely makes your draft return a DIFFERENT value than the user expects on ordinary inputs. Consider, in this order of importance:
1. What exactly the function must compute: its core meaning where the wording is vague, unusual or self-contradictory.
2. The exact form of the result: type (list, tuple, set, str, int, float, bool, ...), structure, order, the exact text of returned strings, and what is returned when there is no result.
3. The exact inputs: number, order and types of the arguments (especially when no signature is given).
4. Conflicts between the text and the examples: which one is right.
Ignore invalid inputs, error handling, input validation, None, and performance unless the task itself makes them central: the tests use valid inputs.

In <analysis>...</analysis>, first write (a) a typical valid call and the exact value your draft returns for it, (b) two or three other values the user might plausibly expect instead and why, and (c) the single decision that distinguishes them.
Then ask about that decision. The question must target exactly one fact, have one objective answer, and {framing} Do not ask about the function name (it is `{entry_point}`), code style, or for test cases{repeat_rule}.

Finish with one line exactly of the form:
QUESTION: <your question>"""

FRAMINGS = {
    "alternatives": 'state your current assumption so the user can confirm or correct it (e.g. "Should it return X (my assumption) or Y?").',
    "open": 'be an open question about that single decision (e.g. "What exactly should it return when ...?" or "How should ... be handled?") that lets the user state the intended behaviour in their own words, instead of offering fixed alternatives.',
    "neutral": 'be phrased neutrally and concisely as a choice between the plausible alternatives (e.g. "Should it return X or Y?"), in plain language without code, and without saying which option you assume.',
}

PRELUDE = (
    "from __future__ import annotations\n"
    "import math\nimport re\nimport string\nimport itertools\nimport functools\n"
    "import collections\nimport heapq\nimport bisect\n"
    "from typing import *\nfrom collections import *\n"
)

FALLBACK = "def {entry_point}(*args, **kwargs):\n    return None\n"

ARITY_WRAPPER = """

def _fs_tolerant(_fn):
    import functools as _ft
    import inspect as _ins
    try:
        _params = list(_ins.signature(_fn).parameters.values())
    except (TypeError, ValueError):
        return _fn
    if not _params or any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in _params):
        return _fn
    _max = len(_params)
    _min = sum(1 for p in _params if p.default is p.empty)

    @_ft.wraps(_fn)
    def _wrapper(*args, **kwargs):
        if not kwargs and len(args) > _max:
            args = args[:_max]
        elif not kwargs and len(args) < _min:
            args = tuple(args) + (None,) * (_min - len(args))
        return _fn(*args, **kwargs)

    return _wrapper


import sys as _fs_sys
_fs_sys.setrecursionlimit(max(_fs_sys.getrecursionlimit(), 4000))
{entry_point} = _fs_tolerant({entry_point})
"""

# ---------------------------------------------------------------------------
# Deterministic code utilities
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
_SIGNATURE_RE = re.compile(r"^\s*def\s+\w+\s*\(", flags=re.MULTILINE)


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _defines(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
        for n in tree.body
    )


def pick_code(response: str, entry_point: str) -> str:
    """Pick the code block that best looks like the solution."""
    blocks = _BLOCK_RE.findall(response or "")
    if not blocks and _parses(response or ""):
        blocks = [response]
    for predicate in (
        lambda b: _parses(b) and _defines(b, entry_point),
        lambda b: _parses(b) and "def " in b,
        lambda b: "def " in b,
    ):
        for block in blocks:
            if predicate(block):
                return block
    return blocks[0] if blocks else ""


def _is_test_code(node, local_names) -> bool:
    """Top-level statement that exercises the code (tests, demos, I/O) rather than defining it."""
    if isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        return False
    if isinstance(node, ast.If) and "__name__" in ast.dump(node.test):
        return True
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return True  # stray docstring / string
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else "")
            )
            if name in {
                "print",
                "input",
                "exit",
                "quit",
                "testmod",
                "main",
                "run",
                "unittest",
            }:
                return True
            if isinstance(fn, ast.Name) and fn.id in local_names:
                return True
    return False


def _prompt_helpers(prompt: str) -> dict[str, str]:
    """Top-level functions *implemented* in the task prompt (e.g. helpers of HumanEval tasks)."""
    try:
        tree = ast.parse(prompt or "")
    except (SyntaxError, ValueError):
        return {}
    helpers = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [
            s
            for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            and not isinstance(s, ast.Pass)
        ]
        if body:
            helpers[node.name] = ast.get_source_segment(prompt, node) or ""
    return helpers


def _free_names(tree) -> set[str]:
    import builtins

    defined, loaded = set(dir(builtins)), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded).add(node.id)
    return loaded - defined


def finalize(code: str, entry_point: str, prompt: str = "", defensive: bool = True) -> str:
    """Make the code safe to prepend to the test harness, and bind `entry_point`."""
    code = (code or "").replace("\r\n", "\n").strip("\n")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        tree = None

    # 1. Drop top-level test/demo code (removal by line ranges keeps the source verbatim).
    if tree is not None:
        local = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        lines = code.split("\n")
        drop = set()
        for node in tree.body:
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            if is_future or _is_test_code(node, local):
                drop.update(range(node.lineno - 1, node.end_lineno))
        code = "\n".join(line for i, line in enumerate(lines) if i not in drop)
        if not _parses(code):
            code = "\n".join(lines)
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            tree = None

    # 2. Bind the entry point if the model used another name (e.g. the name in the prompt).
    if tree is not None and not _defines(code, entry_point):
        funcs = [
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if funcs:
            named = [f for f in funcs if f in re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)]
            called = {
                c.func.id
                for c in ast.walk(tree)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            roots = [f for f in funcs if f not in called]
            code += f"\n\n{entry_point} = {(named or roots or funcs)[-1]}\n"
    if not code.strip() or (
        tree is not None and not _defines(code, entry_point) and f"{entry_point} =" not in code
    ):
        code = (code + "\n\n" if code.strip() else "") + FALLBACK.format(entry_point=entry_point)

    # 3. Restore helpers that the prompt implements and the code calls without defining.
    try:
        missing = _free_names(ast.parse(code))
    except (SyntaxError, ValueError):
        missing = set()
    helpers = _prompt_helpers(prompt)
    extra = [helpers[name] for name in sorted(missing) if name in helpers]
    if extra:
        code = "\n\n".join(extra) + "\n\n" + code

    # 4. Harness robustness: arity tolerance when the task shows no signature; lower-case alias.
    if defensive and _parses(code):
        if not _SIGNATURE_RE.search(prompt or ""):
            code += ARITY_WRAPPER.format(entry_point=entry_point)
        lower = entry_point.lower()
        if lower != entry_point and lower.isidentifier() and not _defines(code, lower):
            code += f"\n{lower} = {entry_point}\n"
    return f"```python\n{PRELUDE}\n{code.strip()}\n```"


def parse_question(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.match(r"^[*`>\-\s]*QUESTION\s*[:：]\s*(.+)$", ln, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("*`").strip()
    tail = (text or "").split("</analysis>")[-1]
    qs = [ln for ln in tail.splitlines() if ln.strip().endswith("?")]
    return qs[-1].strip() if qs else ""


def signature_rule(prompt: str) -> str:
    if _SIGNATURE_RE.search(prompt or ""):
        return " Keep the parameters of the signature shown in the task (only the function name may differ)."
    return " Choose the most natural parameter list for the task (usually the inputs in the order they are mentioned)."


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class ForesightClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # Ask on every task (TDS base 10 makes one turn cost only ~4%); False = never ask.
        "always_ask": True,
        # Maximum number of clarification turns to use (never more than the environment allows).
        "max_questions": 1,
        # Question phrasing: "alternatives" (states the assumption), "open" (targeted open
        # question; TDS-neutral in our tests, higher judged quality), or "neutral".
        "question_framing": "alternatives",
        # Write a short requirement spec before the code.
        "solve_reasoning": False,
        # Harness robustness: arity-tolerant wrapper for signature-less tasks, lower-case alias.
        "defensive_wrappers": True,
        # Wall-clock budget per task (s): no new LLM call starts after it (harness kills at 300 s).
        "time_budget_s": 240,
    }

    def call_llm(self, env: ClarificationEnvironment, content: str, deadline: float) -> str:
        if time.monotonic() > deadline:
            return ""
        for _ in range(2):
            try:
                return env.llm([{"role": "user", "content": content}]) or ""
            except LimitsExceededException:
                return ""
            except Exception:
                continue
        return ""

    def solve(self, env, prompt: str, entry_point: str, pairs, deadline: float) -> str:
        clarification = ""
        if pairs:
            listing = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
            clarification = CLARIFICATION_BLOCK.format(pairs=listing)
        content = SOLVE_TEMPLATE.format(
            entry_point=entry_point,
            prompt=prompt.strip(),
            clarification=clarification,
            signature_rule=signature_rule(prompt),
            output_rule=OUTPUT_RULES[bool(self.config["solve_reasoning"])],
        )
        return pick_code(self.call_llm(env, content, deadline), entry_point)

    def write_question(self, env, prompt, entry_point, draft, pairs, deadline: float) -> str:
        previous = ""
        if pairs:
            listing = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
            previous = f"\nThe user already answered these clarifying questions:\n{listing}\n"
        framing = FRAMINGS.get(self.config["question_framing"], FRAMINGS["alternatives"])
        text = self.call_llm(
            env,
            QUESTION_TEMPLATE.format(
                prompt=prompt.strip(),
                previous=previous,
                entry_point=entry_point,
                candidate=draft or "# (no draft)",
                framing=framing,
                repeat_rule=", nor about anything already answered" if pairs else "",
            ),
            deadline,
        )
        return parse_question(text)

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem.get("entry_point") or "solution"
        prompt = problem.get("prompt") or ""
        defensive = self.config["defensive_wrappers"]
        deadline = time.monotonic() + float(self.config["time_budget_s"])
        draft = ""
        try:
            draft = self.solve(env, prompt, entry_point, [], deadline)
            pairs: list[tuple[str, str]] = []
            while (
                self.config["always_ask"]
                and len(pairs) < int(self.config["max_questions"])
                and env.can_ask()
            ):
                question = self.write_question(env, prompt, entry_point, draft, pairs, deadline)
                if not question:
                    break
                try:
                    answer = (env.ask_human(question) or "").strip()
                except TooManyQuestionException:
                    break
                if not answer or "Cannot answer the given question" in answer:
                    break
                pairs.append((question, answer))
                # Regenerate from scratch: revising the draft anchors on the old reading.
                draft = self.solve(env, prompt, entry_point, pairs, deadline) or draft
            return finalize(draft, entry_point, prompt, defensive)
        except Exception:
            return finalize(draft, entry_point, prompt, defensive)
