# SPDX-FileCopyrightText: 2026 kelvin715 <yzhsk99@gmail.com>
# SPDX-FileCopyrightText: 2026 Dinh <dinhtd.b23cc038@stu.ptit.edu.vn>
#
# SPDX-License-Identifier: MIT
# Contract confirmation wording and execution-table idea adapted from Dinh's
# ContrastiveClarifier, commit a6bd7adff5e1bd556199e3388d3682f3cdb98dfe:
# https://github.com/Dinhh-Chan/clarification-competition/blob/a6bd7adff5e1bd556199e3388d3682f3cdb98dfe/clarify/algorithms/contrastive_clarifier.py

"""Foresight: answer-conditioned verification with retained probe coverage.

Generate three drafts and execute literal probes with strict argument binding.
Ask one question that describes the selected draft's contract and an actual
execution. An explicit confirmation enables multiple final candidates and
answer-bound checks; other usable answers guide one fresh implementation.

When the answer conservatively confirms the executed example, retain earlier
probe inputs alongside fresh probes so boundary coverage is not discarded.
Rank candidates by binding evidence, provisional evidence, return types, and
execution agreement. Accept a repair only if it gains checks without losing a
satisfied binding check. Prompt cleanup permits conservative deletions only;
deterministic finalization restores helpers and the required entry point.

Team: SkyWalker
Team Members: kelvin715
Main Contact: yzhsk99@gmail.com
"""

from __future__ import annotations

import ast
import doctest
import json
import re
import threading
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

FINAL_POLICY = """
Implementation priorities for the clarified task:
- Compute the requested result faithfully. Impossible implementation limits do
  not justify placeholder values or fabricated computation.
- Do not add input-validation exceptions unless explicitly requested. Preserve
  natural, well-defined behavior at zero and empty-input boundaries.
- Keep the clarified return type and format, and follow any explicit error
  behavior. Do not invent new requirements for cases the user did not discuss.
"""

CONTRASTIVE_REQUEST = """[FORESIGHT_CONTRASTIVE_CONTRACT]
Describe the selected implementation's contract so the user can correct its most
important mistaken assumption in one clarification. The first column of the
execution table belongs to the selected draft; other columns are independent
drafts. Pay attention to disagreements on valid, ordinary inputs, especially
the core computation, argument structure, and exact return type and format.
Execution errors are not evidence that a result is correct.

Return only JSON with:
- arguments: the draft's number, order and types of arguments, as a short phrase;
- returns: its exact return type and structure, as a short phrase;
- rule: the computation performed by the selected draft, in one short sentence;
- probe: the zero-based index of a successful execution illustrating its most
  uncertain behavior; prefer an input where successful drafts disagree;
- conflict: null, or {{"a": "verbatim quote", "b": "verbatim quote"}} for two
  prose statements in the task that demand mutually exclusive core behavior.
  Neither quote may be an example or an implementation/style constraint.
Do not propose an alternative answer or ask about names, performance, code style,
or new test cases. Describe what the selected draft actually does.

Task:
{prompt}
Required entry point: {entry}
Selected draft:
```python
{draft}
```
Executed results (selected draft first):
{evidence}
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
{evidence}
You may ask the user exactly ONE clarifying question before finalizing. Pick the uncertainty that most likely makes your draft return a DIFFERENT value than the user expects on ordinary inputs. Consider, in this order of importance:
1. What exactly the function must compute: its core meaning where the wording is vague, unusual or self-contradictory.
2. The exact form of the result: type (list, tuple, set, str, int, float, bool, ...), structure, order, the exact text of returned strings, and what is returned when there is no result.
3. The exact inputs: number, order and types of the arguments (especially when no signature is given).
4. Conflicts between the text and the examples: which one is right.
Ignore invalid inputs, error handling, input validation, None, and performance unless the task itself makes them central: the tests use valid inputs.

In <analysis>...</analysis>, first write (a) a typical valid call and the exact value your draft returns for it, (b) two or three other values the user might plausibly expect instead and why, and (c) the single decision that distinguishes them.
Then ask about that decision. The question must target exactly one fact, have one objective answer, and {framing} Do not ask about the function name (it is `{entry_point}`), code style, or for test cases{repeat_rule}.{plain_rule}

Finish with one line exactly of the form:
QUESTION: <your question>"""

EVIDENCE_NOTE = """
Running your draft on the worked examples shown in the task, these do not match:
{lines}
Either your draft misreads the task or those examples are wrong (task descriptions can contain mistakes).
"""

EXAMPLE_CHECK = """
import ast as ast_7q
import math as math_7q

def same_7q(got_7q, want_7q):
    if isinstance(got_7q, float) and isinstance(want_7q, (int, float)) and not isinstance(want_7q, bool):
        return math_7q.isclose(got_7q, want_7q, rel_tol=1e-6, abs_tol=1e-9)
    return got_7q == want_7q

if "candidate" not in globals() and {entry_point!r} in globals():
    candidate = globals()[{entry_point!r}]

for i_7q, (src_7q, want_7q) in enumerate({cases!r}):
    try:
        got_7q = eval(src_7q)
        tag_7q = "OK" if same_7q(got_7q, ast_7q.literal_eval(want_7q)) else "BAD"
        print("EX7Q", i_7q, tag_7q, repr(got_7q)[:200], flush=True)
    except Exception as e_7q:
        print("EX7Q", i_7q, "ERR", repr(e_7q)[:200], flush=True)
"""

PLAIN_RULE = " Write the question as plain text: no Markdown, no backticks."

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


def _clean_line(line: str) -> str:
    return line.strip().strip("*`>#_-").strip()


def parse_question(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"^[*`>#\-\s]*QUESTION[*`\s]*[:：][*`\s]*(.*)$", lines[i], flags=re.IGNORECASE)
        if not m:
            continue
        question = _clean_line(m.group(1))
        # Markdown header style ("**Question:**" alone): the question is on the next line.
        for follow in lines[i + 1 :]:
            if question:
                break
            if not follow.startswith("<"):
                question = _clean_line(follow)
        if question:
            return question
    tail = (text or "").split("</analysis>")[-1]
    qs = [_clean_line(ln) for ln in tail.splitlines() if _clean_line(ln).endswith("?")]
    return qs[-1] if qs else ""


def plain_text(question: str) -> str:
    """Backtick-free, single-line question (a backtick in the reply truncates the answer)."""
    return " ".join((question or "").replace("`", "").split()).replace("[/", "[ /")


def markup_safe(text: str) -> str:
    return (text or "").replace("[/", "[ /")


def worked_examples(prompt: str) -> list[tuple[str, str]]:
    """The task's own `>>>` examples whose expected output is a Python literal."""
    try:
        examples = doctest.DocTestParser().get_examples(prompt or "")
    except Exception:
        return []
    found = []
    for example in examples:
        source, want = example.source.strip(), example.want.strip()
        for quote in ('"""', "'''"):
            if want.endswith(quote):
                want = want[: -len(quote)].rstrip()
        if not source or not want:
            continue
        try:
            ast.parse(source, mode="eval")
            ast.literal_eval(want)
        except Exception:
            continue
        found.append((source, want))
    return found


def mismatch_lines(output: str, cases: list[tuple[str, str]]) -> list[str]:
    lines = []
    for raw in (output or "").splitlines():
        parts = raw.split(" ", 3)
        if len(parts) < 3 or parts[0] != "EX7Q" or not parts[1].isdigit():
            continue
        index, tag = int(parts[1]), parts[2]
        if tag == "OK" or index >= len(cases):
            continue
        got = parts[3] if len(parts) > 3 else ""
        if tag == "ERR" and got.startswith("NameError"):
            continue
        source, want = cases[index]
        verb = "raises" if tag == "ERR" else "returns"
        lines.append(f"- `{source}`: the task shows `{want}`, your draft {verb} `{got}`")
    return lines


def signature_rule(prompt: str) -> str:
    if _SIGNATURE_RE.search(prompt or ""):
        return " Keep the parameters of the signature shown in the task (only the function name may differ)."
    return " Choose the most natural parameter list for the task (usually the inputs in the order they are mentioned)."


# ---------------------------------------------------------------------------
# Shared model calls and configurable clarification baseline
# ---------------------------------------------------------------------------


class _ForesightBase(ClarificationAlgorithmBase):
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
        # Run the task's own `>>>` examples on the draft and show mismatches to the question writer.
        "example_evidence": True,
        # Retry the question call once when no question can be parsed from the reply.
        "question_retry": True,
        # Send the question as plain text: the simulated user mirrors the question's formatting,
        # and the harness cuts its answer at the first backtick (ANSWERS=`...`).
        "plain_questions": True,
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
                # The harness prints every message with rich, which raises on "[/...]" (read as a
                # closing markup tag); retry once with that sequence defused.
                content = markup_safe(content)
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
        if pairs and self.config.get("final_policy", False):
            content += FINAL_POLICY
        return pick_code(self.call_llm(env, content, deadline), entry_point)

    def example_evidence(
        self, env, prompt: str, entry_point: str, draft: str, deadline: float
    ) -> str:
        cases = worked_examples(prompt)
        if not cases or not draft.strip() or time.monotonic() > deadline:
            return ""
        try:
            check = EXAMPLE_CHECK.format(cases=cases[:8], entry_point=entry_point)
            output = env.exec_code(draft + "\n" + check)
        except Exception:
            return ""
        lines = mismatch_lines(output, cases[:8])
        return EVIDENCE_NOTE.format(lines="\n".join(lines)) if lines else ""

    def write_question(self, env, prompt, entry_point, draft, pairs, deadline: float) -> str:
        previous = ""
        if pairs:
            listing = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
            previous = f"\nThe user already answered these clarifying questions:\n{listing}\n"
        framing = FRAMINGS.get(self.config["question_framing"], FRAMINGS["alternatives"])
        evidence = ""
        if self.config["example_evidence"] and not pairs:
            evidence = self.example_evidence(env, prompt, entry_point, draft, deadline)
        content = QUESTION_TEMPLATE.format(
            prompt=prompt.strip(),
            previous=previous,
            entry_point=entry_point,
            candidate=draft or "# (no draft)",
            framing=framing,
            repeat_rule=", nor about anything already answered" if pairs else "",
            evidence=evidence,
            plain_rule=PLAIN_RULE if self.config["plain_questions"] else "",
        )
        question = parse_question(self.call_llm(env, content, deadline))
        if not question and self.config["question_retry"]:
            question = parse_question(self.call_llm(env, content, deadline))
        if self.config["plain_questions"]:
            question = plain_text(question)
        return question

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


# ---------------------------------------------------------------------------
# Evidence prompts and bounded sandbox execution
# ---------------------------------------------------------------------------
# The evidence hierarchy is inspired by the public ContrastiveClarifier and
# CrossCheck submissions (PRs #9 and #4).
PROBE_REQUEST = """[FORESIGHT_PROBES]
Write up to {count} ordinary, valid calls of {entry} for the requirement below.
Use only input types and values the requirement allows. Include informative cases
that distinguish plausible meanings, output formats, or ordering rules. Do not
predict expected outputs. Return JSON: {{"calls": ["{entry}(...)", ...]}}.
Every argument must be a Python literal; keyword arguments are allowed.
Requirement:
{prompt}
Draft:
{draft}
{clarification}
"""

FINAL_PROBE_REQUEST = """[FORESIGHT_PROBES]
Write up to {count} small, ordinary, valid calls of {entry} for the clarified task.
Return JSON: {{"calls": ["{entry}(...)", ...]}}. Every argument must be a Python
literal; keyword arguments are allowed. Do not predict expected outputs.
Task:
{prompt}
Question (context about an earlier interpretation, not additional requirements):
{question}
Answer:
{answer}

Derive the argument count, input types and valid domain from the task and answer.
The answer is authoritative wherever specific. If it corrects the earlier
interpretation, use that corrected interface; do not reuse rejected argument
shapes from the question. Include valid cases that distinguish plausible
computations, output formats or ordering rules. Avoid speculative invalid inputs.
"""

WITNESS_REQUEST = """[FORESIGHT_WITNESS]
Requirement:
{prompt}
Function: {entry}
Draft:
{draft}
Actual execution (these outputs describe the draft, not the user's intention):
{evidence}
You can ask one clarification. Identify the one uncertain fact most likely to
change the draft's observable result on ordinary valid inputs. Prefer core
meaning, exact return format, arguments, or a contradiction over speculative
edge cases. Do not ask for an implementation, test suite, or the whole spec.
Return JSON with keys:
"probe": zero-based index of the executed call that illustrates this uncertainty,
"question": one concise, atomic question that lets the user resolve that fact;
  the actual example will be prepended separately, so do not repeat its output,
"arguments": a short description of the draft's inputs,
"returns": a short description of its actual result type and format,
"rule": one short sentence describing its rule.
Use plain text without backticks. Do not infer that agreeing drafts are correct.
"""

CHECK_REQUEST = """[FORESIGHT_CHECKS]
Requirement:
{prompt}
Question: {question}
User's answer: {answer}
Public examples (zero-based):
{examples}
Executed example shown in the question: {witness}

Extract executable evidence without inventing missing requirements. The answer
overrides the requirement only where it is specific. Return a JSON object:
"confirms_witness": true only if the answer explicitly accepts the exact example,
"confirmation_quote": exact supporting words copied from the answer,
"examples": an array of strings, e.g. ["KEEP", "DROP"], in public-example index
 order; use [] when there are no examples. Each string is KEEP, DROP, or CONFLICT. DROP if the
 answer contradicts it; CONFLICT if it conflicts with the requirement and the
 answer leaves that conflict unresolved; otherwise KEEP,
"checks": up to six objects {{"call": "{entry}(...)", "expected": "Python literal",
 "quote": "exact supporting passage from the answer"}}. Include calls the answer
 states or unambiguously settles. If it settles no exact output, use an empty list.
"types": up to two objects {{"type": "list|tuple|set|dict|str|int|float|bool",
 "quote": "exact passage specifying the return type for all valid inputs"}}.
Do not turn an example of a type into a universal claim. Preserve strings and
container types. An answer saying 'not 3' does not establish 3 as the result.
"""

REPAIR_REQUEST = """[FORESIGHT_REPAIR]
Implement {entry} for this requirement:
{prompt}
Question: {question}
Answer: {answer}
Current implementation:
{code}
Observed failing checks:
{failures}
Checks already satisfied:
{passing}
Fix the demonstrated failures while preserving all other behavior. The actual
answer overrides conflicting prompt text. Do not special-case the shown inputs;
implement the general rule. Do not add unrequested validation or change types.
Return one complete python code block, with imports, and no tests or demo calls.
"""

EVIDENCE_HARNESS = r"""
import ast, contextlib, copy, hashlib, inspect, io, json, math, signal, time

def timeout_handler(signum, frame):
    raise TimeoutError("evidence timeout")

signal.signal(signal.SIGALRM, timeout_handler)

def bounded(fn, seconds):
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def same(actual, expected):
    try:
        if actual == expected:
            return True
    except Exception:
        pass
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, float) or isinstance(expected, float):
            return math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-6)
    if isinstance(actual, (list, tuple)) and type(actual) is type(expected):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return False

def execute_call(fn, call, budget, strict=False):
    args, kwargs = ast.literal_eval(call[0]), ast.literal_eval(call[1])
    def invoke():
        if strict:
            try:
                signature = inspect.signature(fn)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                # functools.wraps exposes the raw draft's signature. A delivery
                # adapter must not manufacture a successful probe by dropping
                # extra arguments or inserting missing arguments.
                signature.bind(*args, **kwargs)
        return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    return bounded(invoke, budget)

reports = []
share = max(0.1, TOTAL_SECONDS / max(1, len(CODES)))
for code in CODES:
    deadline = time.monotonic() + share
    row = {"load": False, "checks": [], "probes": [], "types": 0}
    namespace = {"__name__": "foresight_candidate"}
    try:
        bounded(lambda: exec(code, namespace), min(2.0, share))
        fn = namespace[ENTRY]
        if not callable(fn):
            raise TypeError("entry is not callable")
        row["load"] = True
    except BaseException:
        reports.append(row)
        continue
    for check in CHECKS:
        if time.monotonic() >= deadline:
            row["checks"].append({"ok": None, "got": "not run"})
            continue
        try:
            out = execute_call(fn, check["call"], min(0.75, deadline-time.monotonic()))
            row["checks"].append({"ok": bool(same(out, ast.literal_eval(check["expected"]))), "got": repr(out)[:240]})
        except BaseException as error:
            row["checks"].append({"ok": False, "got": type(error).__name__})
    for call in CALLS:
        if time.monotonic() >= deadline:
            row["probes"].append({"ok": False, "got": "not run"})
            continue
        try:
            out = execute_call(fn, call, min(0.75, deadline-time.monotonic()), strict=STRICT_PROBES)
            shown = repr(out)
            fingerprint = hashlib.sha256((type(out).__name__ + ":" + shown).encode()).hexdigest()
            row["probes"].append({"ok": True, "got": shown[:240], "digest": fingerprint})
            row["types"] += sum(type(out).__name__ == name for name in TYPES)
        except BaseException as error:
            row["probes"].append({"ok": False, "got": type(error).__name__})
    reports.append(row)
print("FORESIGHT_EVIDENCE=" + json.dumps(reports))
"""


# ---------------------------------------------------------------------------
# Literal evidence parsing, answer interpretation, and candidate selection
# ---------------------------------------------------------------------------


def read_json_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def literal_call(source, names):
    """Parse data only. Generated expressions are never executed in the algorithm."""
    if not isinstance(source, str) or len(source) > 4000:
        return None
    try:
        node = ast.parse(source.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        if node.func.id not in names or any(k.arg is None for k in node.keywords):
            return None
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        return (repr(args), repr(kwargs))
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return None


def display_call(entry, call):
    try:
        args, kwargs = ast.literal_eval(call[0]), ast.literal_eval(call[1])
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        return entry + "(" + ", ".join(parts) + ")"
    except (ValueError, SyntaxError, TypeError):
        return entry + "(...)"


def literal_prefix(text):
    """Accept a whole literal followed by prose punctuation, never an arithmetic prefix."""
    line = text.strip().split("\n", 1)[0]
    if not line or len(line) > 2000:
        return None
    for end in range(len(line), 0, -1):
        piece = line[:end].strip()
        rest = line[end:].strip()
        if rest and not (rest.startswith((";", ". ", ", ")) or rest == "."):
            continue
        try:
            # A sentence's final period is punctuation, not a float suffix.
            return repr(ast.literal_eval(piece.removesuffix(".")))
        except (ValueError, SyntaxError, TypeError, RecursionError):
            pass
    return None


def literal_examples(text, names):
    """Extract only explicit literal calls/results, including doctest and inline notation."""
    cases = []
    for source, expected in worked_examples(text):
        call = literal_call(source, names)
        if call:
            cases.append({"call": call, "expected": repr(ast.literal_eval(expected))})
    names_pattern = "|".join(re.escape(name) for name in names)
    for match in re.finditer(r"\b(?:" + names_pattern + r")\s*\(", text or ""):
        start = match.start()
        tail = text[start:]
        for end in range(match.end() - start, min(len(tail), 2000)):
            if tail[end] != ")":
                continue
            call = literal_call(tail[: end + 1], names)
            if not call:
                continue
            rest = tail[end + 1 :]
            connector = re.match(
                r"\s*(?:==|=>|->|returns?|should\s+return|must\s+return)\s*:?\s*", rest
            )
            if connector:
                expected = literal_prefix(rest[connector.end() :])
                if expected is not None:
                    cases.append({"call": call, "expected": expected})
            break
    unique = {}
    for case in cases:
        unique.setdefault(case_key(case), case)
    return list(unique.values())


def case_key(case):
    return (tuple(case["call"]), case["expected"])


def supported_quote(quote, answer):
    if not isinstance(quote, str) or len(quote.strip()) < 3:
        return False
    return " ".join(quote.split()) in " ".join(answer.split())


def confirmed_witness(value, answer, witness):
    """Require a quoted confirmation without correction markers."""
    confirmation = value.get("confirmation_quote", "")
    return bool(
        witness
        and value.get("confirms_witness") is True
        and supported_quote(confirmation, answer)
        and re.search(r"\b(yes|correct|right|exactly|confirmed|indeed)\b", confirmation, re.I)
        and not re.search(r"\b(not|but|except|instead|however|rather)\b", answer, re.I)
    )


def needs_answer_checks(config, answer, witness):
    """Allocate candidate verification using only the answer and executed witness."""
    if not config["answer_checks"]:
        return False
    if not config["checks_on_confirmation_only"]:
        return True
    # Confirmation leaves the current interpretation in place: spend extra
    # computation on boundary coverage. A corrected contract gets the normal
    # fresh implementation, whose answer-conditioned prompt is unchanged.
    return bool(
        witness
        and re.search(r"^\s*(yes|correct|exactly|indeed|confirmed)\b", answer, re.I)
        and not re.search(r"\b(not|but|except|instead|however|rather)\b", answer, re.I)
    )


def answer_probes(fresh, previous, value, answer, witness, limit):
    """Retain earlier inputs only when the answer confirms the executed example."""
    calls = fresh or previous
    if confirmed_witness(value, answer, witness):
        # A positive confirmation should not discard earlier coverage. Keep
        # previous probes only on this conservative, unchanged-contract path.
        calls = list(dict.fromkeys([*calls, *previous]))[: 2 * limit]
    return calls


def compile_evidence(value, answer, examples, witness, names, max_checks):
    """Keep direct answer evidence even when every generated program disagrees."""
    hard = literal_examples(answer, names)
    # A literal mentioned in a denial or hypothetical is not an asserted result.
    # The compiler can still propose it as provisional evidence with a quote.
    if re.search(r"\b(not|wrong|incorrect|would|might|hypothetical|if)\b", answer, re.IGNORECASE):
        hard = []
    expected_by_call = {}
    for case in hard:
        expected_by_call.setdefault(tuple(case["call"]), set()).add(case["expected"])
    hard = [c for c in hard if len(expected_by_call[tuple(c["call"])]) == 1]
    for case in hard:
        case["hard"] = True
    if confirmed_witness(value, answer, witness):
        hard.append({**witness, "hard": True})
    soft = []
    verdicts = value.get("examples", [])
    if isinstance(verdicts, dict):
        verdicts = [verdicts.get(str(i), "KEEP") for i in range(len(examples))]
    elif isinstance(verdicts, str) and len(examples) == 1:
        verdicts = [verdicts]
    elif not isinstance(verdicts, list):
        verdicts = []
    for i, case in enumerate(examples):
        verdict = str(verdicts[i]).upper() if i < len(verdicts) else "KEEP"
        if verdict in {"KEEP", "CONFIRMED"}:
            soft.append({**case, "hard": False})
    proposed = value.get("checks", [])
    for case in proposed[:max_checks] if isinstance(proposed, list) else []:
        if not isinstance(case, dict) or not supported_quote(case.get("quote"), answer):
            continue
        call = literal_call(case.get("call"), names)
        try:
            expected_value = case.get("expected", "")
            # The compiler sometimes emits a JSON value instead of the
            # requested serialized Python literal. Normalize both data-only
            # forms; a string still follows the documented literal format.
            source = expected_value if isinstance(expected_value, str) else repr(expected_value)
            expected = repr(ast.literal_eval(source))
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
        if call:
            soft.append({"call": call, "expected": expected, "hard": False})
    # The same call may have a corrupted public answer. An explicit answer wins.
    hard_calls = {tuple(case["call"]) for case in hard}
    unique = {case_key(c): c for c in hard}
    for case in soft:
        if tuple(case["call"]) not in hard_calls:
            unique.setdefault(case_key(case), case)
    types = []
    for item in value.get("types", []) if isinstance(value.get("types"), list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("type")
        if name in {"list", "tuple", "set", "dict", "str", "int", "float", "bool"}:
            if supported_quote(item.get("quote"), answer) and name not in types:
                types.append(name)
    return list(unique.values())[:max_checks], types


def filter_provisional(checks, rows):
    """Discard unsupported provisional checks while preserving binding evidence."""
    live = [r for r in rows if r.get("load")]
    if len(live) < 2:
        return checks, rows
    keep = []
    for i, check in enumerate(checks):
        complete = all(i < len(r["checks"]) and r["checks"][i]["ok"] is not None for r in live)
        if check["hard"] or not complete or any(r["checks"][i]["ok"] for r in live):
            keep.append(i)
    return [checks[i] for i in keep], [
        {**r, "checks": [r["checks"][i] for i in keep]} if r.get("load") else r for r in rows
    ]


def evidence_rank(rows, checks):
    """Order evidence by authority before using execution agreement to break ties."""

    def signature(row):
        probes = row.get("probes", [])
        if not probes or any(not p.get("ok") for p in probes):
            return None
        return tuple(p["digest"] for p in probes)

    signatures = [signature(row) for row in rows]

    def key(i):
        row = rows[i]
        if not row.get("load"):
            return (-1, -1, -1, -1, -i)
        flags = [c.get("ok") is True for c in row.get("checks", [])]
        hard = sum(flags[j] for j, c in enumerate(checks) if c["hard"] and j < len(flags))
        soft = sum(flags[j] for j, c in enumerate(checks) if not c["hard"] and j < len(flags))
        support = signatures.count(signatures[i]) if signatures[i] is not None else 0
        return (hard, soft, row.get("types", 0), support, -i)

    return max(range(len(rows)), key=key)


def improved_without_regression(old, new, checks):
    """Accept strictly more satisfied checks without losing a binding success."""
    if not new.get("load") or len(new.get("checks", [])) != len(checks):
        return False
    if len(old.get("checks", [])) != len(checks):
        return False
    before = [c.get("ok") is True for c in old["checks"]]
    after = [c.get("ok") is True for c in new["checks"]]
    if any(c["hard"] and before[i] and not after[i] for i, c in enumerate(checks)):
        return False
    return sum(after) > sum(before)


# ---------------------------------------------------------------------------
# Single-clarification evidence pipeline
# ---------------------------------------------------------------------------


class _EvidenceForesight(_ForesightBase):
    DEFAULT_CONFIG = {
        **_ForesightBase.DEFAULT_CONFIG,
        "question_mode": "witness",  # original / witness / contract / contrastive
        "answer_checks": True,
        "checks_on_confirmation_only": False,
        "draft_samples": 1,
        "final_samples": 3,
        "probe_count": 8,
        "max_checks": 12,
        "repair_rounds": 1,
        "exec_seconds": 20.0,
        "batch_timeout_s": 120.0,
        "final_policy": False,
        "strict_probe_signature": True,
    }

    def parallel_prompts(self, env, prompts, deadline):
        results = [""] * len(prompts)

        def call(index, prompt):
            results[index] = self.call_llm(env, prompt, deadline)

        stop = min(deadline, time.monotonic() + float(self.config["batch_timeout_s"]))
        threads = [
            threading.Thread(target=call, args=(i, p), daemon=True) for i, p in enumerate(prompts)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, stop - time.monotonic()))
        return list(results)

    def code_prompt(self, prompt, entry, pairs):
        block = (
            CLARIFICATION_BLOCK.format(pairs="\n".join(f"Q: {q}\nA: {a}" for q, a in pairs))
            if pairs
            else ""
        )
        content = SOLVE_TEMPLATE.format(
            entry_point=entry,
            prompt=prompt.strip(),
            clarification=block,
            signature_rule=signature_rule(prompt),
            output_rule=OUTPUT_RULES[False],
        )
        if pairs and self.config["final_policy"]:
            content += FINAL_POLICY
        return content

    def executable(self, code, entry, prompt):
        return pick_code(finalize(code, entry, prompt, self.config["defensive_wrappers"]), entry)

    def execute_evidence(self, env, codes, entry, prompt, calls, checks, types, deadline):
        if not codes or time.monotonic() >= deadline:
            return None
        values = {
            "CODES": [self.executable(c, entry, prompt) for c in codes],
            "ENTRY": entry,
            "CALLS": calls,
            "CHECKS": checks,
            "TYPES": types,
            "STRICT_PROBES": bool(self.config["strict_probe_signature"]),
            "TOTAL_SECONDS": max(
                0.1, min(float(self.config["exec_seconds"]), deadline - time.monotonic())
            ),
        }
        script = "\n".join(f"{k} = {v!r}" for k, v in values.items()) + "\n" + EVIDENCE_HARNESS
        try:
            output = env.exec_code(script) or ""
            marked = output.rsplit("FORESIGHT_EVIDENCE=", 1)[1].splitlines()[0]
            rows = json.loads(marked)
            if len(rows) != len(codes) or any(not isinstance(r, dict) for r in rows):
                return None
            return rows
        except Exception:
            return None

    def probes(self, reply, names):
        value = read_json_object(reply).get("calls", [])
        calls = []
        for source in value if isinstance(value, list) else []:
            call = literal_call(source, names)
            if call and call not in calls:
                calls.append(call)
        return calls[: int(self.config["probe_count"])]

    def contrastive_question(self, env, prompt, entry, draft, rows, calls, deadline):
        rows = rows or []
        lines = []
        for i, call in enumerate(calls):
            probes = [row.get("probes", [])[i] for row in rows if i < len(row.get("probes", []))]
            outputs = [p.get("got", "?") if p.get("ok") else "execution error" for p in probes]
            successful = {p.get("got", "?") for p in probes if p.get("ok")}
            marker = " (drafts disagree)" if len(successful) > 1 else ""
            lines.append(f"{i}: {display_call(entry, call)} -> " + " | ".join(outputs) + marker)
        value = read_json_object(
            self.call_llm(
                env,
                CONTRASTIVE_REQUEST.format(
                    prompt=prompt,
                    entry=entry,
                    draft=draft,
                    evidence="\n".join(lines) or "No reliable execution available.",
                ),
                deadline,
            )
        )
        conflict = value.get("conflict")
        if isinstance(conflict, dict):
            a, b = conflict.get("a"), conflict.get("b")
            if (
                isinstance(a, str)
                and isinstance(b, str)
                and a != b
                and 8 <= len(a.strip()) <= 180
                and 8 <= len(b.strip()) <= 180
                and a in prompt
                and b in prompt
                and not any(token in a + b for token in (">>>", "```"))
            ):
                return plain_text(
                    f'About {entry}: the task says "{a}" but also "{b}". Which is intended?'
                ), None
        if not all(
            isinstance(value.get(k), str) and value[k].strip()
            for k in ("arguments", "returns", "rule")
        ):
            return self.write_question(env, prompt, entry, draft, [], deadline), None
        probes = rows[0].get("probes", []) if rows else []
        try:
            requested = int(value.get("probe", -1))
        except (ValueError, TypeError):
            requested = -1
        order = [requested] if 0 <= requested < len(probes) else []
        order += [i for i in range(min(len(calls), len(probes))) if i not in order]
        witness = None
        for i in order:
            probe = probes[i]
            if i >= len(calls) or not probe.get("ok"):
                continue
            try:
                expected = repr(ast.literal_eval(probe["got"]))
                if len(expected) <= 120:
                    witness = {"call": calls[i], "expected": expected}
                    break
            except (ValueError, SyntaxError, TypeError, KeyError):
                continue
        question = (
            f"About {entry}: my draft takes {value['arguments'][:220]} and returns "
            f"{value['returns'][:220]}; {value['rule'][:300]}."
        )
        if witness:
            question += f" For example, {display_call(entry, witness['call'])} returns {witness['expected']}."
        question += " Is that what you intend, or should it behave differently or return a different format?"
        return plain_text(question), witness

    def witnessed_question(self, env, prompt, entry, draft, rows, calls, deadline):
        if self.config["question_mode"] == "contrastive":
            return self.contrastive_question(env, prompt, entry, draft, rows, calls, deadline)
        row = rows[0] if rows else {}
        lines = [
            f"{i}: {display_call(entry, call)} -> {p.get('got')}"
            for i, (call, p) in enumerate(zip(calls, row.get("probes", []), strict=False))
            if p.get("ok")
        ]
        value = read_json_object(
            self.call_llm(
                env,
                WITNESS_REQUEST.format(
                    prompt=prompt,
                    entry=entry,
                    draft=draft,
                    evidence="\n".join(lines) or "No reliable execution available.",
                ),
                deadline,
            )
        )
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            return self.write_question(env, prompt, entry, draft, [], deadline), None
        witness = None
        try:
            index = int(value.get("probe", -1))
            probe = row.get("probes", [])[index] if index >= 0 else {}
            if probe.get("ok") and index < len(calls):
                expected = repr(ast.literal_eval(probe["got"]))
                witness = {"call": calls[index], "expected": expected}
        except (ValueError, SyntaxError, TypeError, IndexError):
            pass
        if self.config["question_mode"] == "contract" and all(
            isinstance(value.get(k), str) and value[k].strip()
            for k in ("arguments", "returns", "rule")
        ):
            question = (
                f"My draft takes {value['arguments']}, returns {value['returns']}, and {value['rule']}. "
                "Which one part of this behavior, if any, should change?"
            )
        if witness:
            question = (
                f"My current implementation returns {witness['expected']} for {display_call(entry, witness['call'])}. "
                + question
            )
        return plain_text(question), witness

    def run(self, env, problem):
        if self.config["question_mode"] == "original" and not self.config["answer_checks"]:
            return super().run(env, problem)
        prompt, entry = problem.get("prompt", ""), problem.get("entry_point") or "solution"
        deadline = time.monotonic() + float(self.config["time_budget_s"])
        names = {entry, "candidate"} | set(re.findall(r"\bdef\s+(\w+)\s*\(", prompt))
        code = ""
        try:
            calls, witness = [], None
            if self.config["question_mode"] == "original":
                code = self.solve(env, prompt, entry, [], deadline)
                question = self.write_question(env, prompt, entry, code, [], deadline)
            else:
                draft_prompt = self.code_prompt(prompt, entry, [])
                n = max(1, int(self.config["draft_samples"]))
                replies = self.parallel_prompts(
                    env,
                    [draft_prompt] * n
                    + [
                        PROBE_REQUEST.format(
                            count=self.config["probe_count"],
                            entry=entry,
                            prompt=prompt,
                            draft="",
                            clarification="",
                        )
                    ],
                    deadline,
                )
                drafts = [
                    c
                    for c in (pick_code(r, entry) for r in replies[:n])
                    if _parses(c) and c.strip()
                ]
                code = drafts[0] if drafts else ""
                calls = self.probes(replies[-1], names)
                public = [
                    {**c, "hard": False}
                    for c in literal_examples(prompt, names)[: int(self.config["max_checks"])]
                ]
                rows = self.execute_evidence(
                    env, drafts, entry, prompt, calls, public, [], deadline
                )
                if rows:
                    selected = evidence_rank(rows, public)
                    code = drafts[selected]
                    order = [selected] + [i for i in range(len(rows)) if i != selected]
                    rows = [rows[i] for i in order]
                question, witness = self.witnessed_question(
                    env, prompt, entry, code, rows, calls, deadline
                )
            if not question or not self.config["always_ask"] or not env.can_ask():
                return finalize(code, entry, prompt, self.config["defensive_wrappers"])
            try:
                answer = (env.ask_human(question) or "").strip()
            except Exception:
                answer = ""
            if not answer or "Cannot answer the given question" in answer:
                return finalize(code, entry, prompt, self.config["defensive_wrappers"])
            if not needs_answer_checks(self.config, answer, witness):
                code = self.solve(env, prompt, entry, [(question, answer)], deadline) or code
                return finalize(code, entry, prompt, self.config["defensive_wrappers"])
            examples = literal_examples(prompt, names)[: int(self.config["max_checks"])]
            numbered = "\n".join(
                f"{i}: {display_call(entry, c['call'])} -> {c['expected']}"
                for i, c in enumerate(examples)
            )
            n = max(1, int(self.config["final_samples"]))
            prompts = [self.code_prompt(prompt, entry, [(question, answer)])] * n
            prompts += [
                CHECK_REQUEST.format(
                    prompt=prompt,
                    question=question,
                    answer=answer,
                    examples=numbered or "none",
                    entry=entry,
                    witness=(display_call(entry, witness["call"]) + " -> " + witness["expected"])
                    if witness
                    else "none",
                ),
                FINAL_PROBE_REQUEST.format(
                    count=self.config["probe_count"],
                    entry=entry,
                    prompt=prompt,
                    question=question,
                    answer=answer,
                ),
            ]
            replies = self.parallel_prompts(env, prompts, deadline)
            finals = [
                c for c in (pick_code(r, entry) for r in replies[:n]) if _parses(c) and c.strip()
            ]
            if not finals:
                return finalize(code, entry, prompt, self.config["defensive_wrappers"])
            code = finals[0]
            compiled = read_json_object(replies[n])
            checks, types = compile_evidence(
                compiled,
                answer,
                examples,
                witness,
                names,
                int(self.config["max_checks"]),
            )
            calls = answer_probes(
                self.probes(replies[n + 1], names),
                calls,
                compiled,
                answer,
                witness,
                int(self.config["probe_count"]),
            )
            rows = self.execute_evidence(env, finals, entry, prompt, calls, checks, types, deadline)
            if not rows:
                return finalize(code, entry, prompt, self.config["defensive_wrappers"])
            checks, rows = filter_provisional(checks, rows)
            selected = evidence_rank(rows, checks)
            code, status = finals[selected], rows[selected]
            for _ in range(int(self.config["repair_rounds"])):
                results = status.get("checks", [])
                if (
                    not checks
                    or len(results) != len(checks)
                    or all(c.get("ok") is True for c in results)
                ):
                    break
                failing, passing = [], []
                for check, result in zip(checks, results, strict=True):
                    description = f"{display_call(entry, check['call'])} -> {check['expected']}"
                    if result.get("ok") is True:
                        passing.append(description)
                    elif result.get("ok") is False:
                        failing.append(description + "; observed " + result.get("got", "?"))
                if not failing:
                    break
                repaired = pick_code(
                    self.call_llm(
                        env,
                        REPAIR_REQUEST.format(
                            entry=entry,
                            prompt=prompt,
                            question=question,
                            answer=answer,
                            code=code,
                            failures="\n".join(failing),
                            passing="\n".join(passing) or "none",
                        ),
                        deadline,
                    ),
                    entry,
                )
                if not repaired or not _parses(repaired):
                    break
                trial = self.execute_evidence(
                    env, [repaired], entry, prompt, [], checks, [], deadline
                )
                if not trial or not improved_without_regression(status, trial[0], checks):
                    break
                code, status = repaired, trial[0]
        except Exception:
            pass
        return finalize(code, entry, prompt, self.config["defensive_wrappers"])


# ---------------------------------------------------------------------------
# Conservative prompt cleanup
# ---------------------------------------------------------------------------

CLEANUP_REQUEST = """[FORESIGHT_CLEANUP]
Inspect this Python task for internally contradictory or impossible add-on clauses.
Preserve its core computational intent, function signature, valid constraints,
examples, and all underspecified details. Do not solve it or fill in missing facts.
You may only DELETE exact verbatim spans. Delete a span only when you can name a
concrete conflict with the rest of the task (such as an impossible complexity
requirement, incompatible input/output types, or two mutually exclusive rules).
Do not delete a clause just because it is unusual or inconvenient. Do not delete
imports, function signatures, helper implementations, or worked examples. If no
clear contradiction exists, return an empty list. Never replace or add text.
Return JSON: {{"deletions": [{{"span": "exact text", "reason": "specific conflict"}}]}}.
Task:
{prompt}
"""


def code_without_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return ast.dump(tree, include_attributes=False)


def deletion_bounds(text, requested):
    """Locate a unique source span; only whitespace may differ in the request."""
    exact = text.count(requested)
    if exact == 1:
        start = text.index(requested)
        return start, start + len(requested)
    if exact > 1:
        return None
    tokens = list(re.finditer(r"\w+|[^\w\s]", text))
    needle = re.findall(r"\w+|[^\w\s]", requested)
    if not needle or len(needle) > len(tokens):
        return None
    values = [token.group() for token in tokens]
    matches = []
    for start in range(len(tokens) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            matches.append((tokens[start].start(), tokens[start + len(needle) - 1].end()))
            if len(matches) > 1:
                return None
    return matches[0] if matches else None


def deletion_only_cleanup(prompt, reply):
    """Delete unique original spans while preserving code and worked examples."""
    items = read_json_object(reply).get("deletions", [])
    if not isinstance(items, list):
        return prompt
    cleaned = prompt
    original_tokens = len(re.findall(r"\w+|[^\w\s]", prompt))
    for item in items[:8]:
        if not isinstance(item, dict) or not isinstance(item.get("reason"), str):
            continue
        span = item.get("span")
        if not isinstance(span, str) or len(span.strip()) < 8:
            continue
        bounds = deletion_bounds(cleaned, span)
        if bounds is None:
            continue
        start, end = bounds
        original_span = cleaned[start:end]
        if re.search(r"\b(def|class|import|from)\s|>>>|\.\.\.|\"\"\"|'''", original_span):
            continue
        line_start = cleaned.rfind("\n", 0, start) + 1
        # A deletion within a code/example line could silently change its meaning.
        line_prefix = cleaned[line_start:start]
        if re.search(r"\b(def|class|import|from)\s|>>>", line_prefix):
            continue
        trial = cleaned[:start] + cleaned[end:]
        if len(re.findall(r"\w+|[^\w\s]", trial)) < 0.6 * original_tokens:
            continue
        # Doctest source and expected output must remain byte-for-byte equivalent.
        if worked_examples(trial) != worked_examples(prompt):
            continue
        if _parses(prompt):
            if not _parses(trial) or code_without_docstrings(trial) != code_without_docstrings(
                prompt
            ):
                continue
        cleaned = trial
    return cleaned


# ---------------------------------------------------------------------------
# Public submission: the defaults used for the complete validation run
# ---------------------------------------------------------------------------


class ForesightClarification(_EvidenceForesight):
    """Compose conservative cleanup with answer-conditioned verification."""

    DEFAULT_CONFIG = {
        **_EvidenceForesight.DEFAULT_CONFIG,
        "normalize_prompt": True,
        "question_mode": "contrastive",
        "draft_samples": 3,
        "answer_checks": True,
        "checks_on_confirmation_only": True,
        "final_policy": True,
    }

    def run(self, env, problem):
        started = time.monotonic()
        prompt = problem.get("prompt") or ""
        if self.config["normalize_prompt"]:
            reply = self.call_llm(
                env,
                CLEANUP_REQUEST.format(prompt=prompt),
                started + min(60.0, float(self.config["time_budget_s"])),
            )
            prompt = deletion_only_cleanup(prompt, reply)
        config = {k: v for k, v in self.config.items() if k in _EvidenceForesight.DEFAULT_CONFIG}
        config["time_budget_s"] = max(
            0.0, float(config["time_budget_s"]) - (time.monotonic() - started)
        )
        return _EvidenceForesight(config).run(env, {**problem, "prompt": prompt})
