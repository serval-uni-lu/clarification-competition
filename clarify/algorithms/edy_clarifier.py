# SPDX-FileCopyrightText: 2026 杨嵩
#
# SPDX-License-Identifier: MIT

"""EDYClarifier.

Draft-anchored, always-ask clarification with execution-checked synthesis.

1. Draft: write a first implementation so the analysis is grounded in concrete
   behaviour (and, for signature-less MBPP tasks, a concrete interface).
2. Probe: analyse contradictions, interface gaps and vague terms against the
   draft, and ask exactly one atomic, binary question about the decision that
   flips the most test outcomes. One question costs ~4% TDS, which is far less
   than a missed hidden requirement, so the algorithm always asks.
3. Solve fresh: regenerate from scratch with the answer marked authoritative
   (revising the draft anchors the model to its first reading). Unresolved
   noise is resolved toward the conventional behaviour of the function.
4. Check: run the prompt's own examples in the sandbox; on failures, a
   reconciliation step decides whether the example or the clarification wins.
5. Sanitize: strip test/demo code, bind the entry point, add common imports.

Team: EDY
Team Members: 曹向前 (Cao Xiangqian, Inner Mongolia University), 杨嵩 (Yang Song, Suzhou Heshuju Information Technology Co., Ltd.), 李凯 (Li Kai, Suzhou Heshuju Information Technology Co., Ltd.)
Main Contact: 杨嵩(Yang Song) <song.yang@core-dt.com>
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

from rich.markup import escape

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

PRIOR_NOTE = """
Background: tasks like this one are (possibly perturbed) versions of well-known short
Python exercises in the style of MBPP and HumanEval, and the hidden tests follow the
original exercise. Where the text is vague, incomplete or contradictory and nothing
else settles it, behave as the canonical exercise would, including its usual call
signature (for example extra length or index arguments) and its usual return format.
"""

DRAFT_TEMPLATE = """
Write a self-contained Python implementation of `{entry_point}` for the task below.
{prior}
Task:
{prompt}

Pick the most natural interpretation and the most likely call signature.
Enclose the code in ```python and ```. Do not include tests or example usage.
""".strip()

QUESTION_TEMPLATE = """
A user gave you a coding task. The task text may be underspecified, vague, or may even
contradict itself or its own examples on purpose. The user has hidden requirements and
a hidden test suite; you may ask the user exactly ONE clarifying question before the
final implementation is written.

Task (function `{entry_point}`):
{prompt}

Your first draft:
```python
{draft}
```

Find the single decision whose wrong guess would make the most hidden tests fail.
Check, in this priority order:
1. Contradictions: two statements (or a statement and an example) that cannot both
   hold, or a requirement that is impossible or nonsensical. Ask which side is meant.
2. Interface: {interface_hint}
3. Core semantics: a vague or overloaded term, an unspecified rule (inclusive vs
   exclusive, strict vs non-strict, ordering, tie-breaking, what is counted or
   returned), or an unusual output format the task hints at.
4. Edge behaviour that the tests are likely to hit (empty input, no result found).

Rules for the question:
- It asks about exactly one fact and has one short, objective answer.
- Prefer a binary choice that states your assumption, e.g.
  "Should X be done (my assumption), or should Y be done instead?"
  A concrete input with the two candidate outputs makes it sharper.
- Never ask for the algorithm, code, test cases, or several things at once.
- Do not use backticks, bullet points, or numbering. One sentence, one question mark.
- Start directly with the question itself (e.g. "Should ...", "Does ...", "When ...");
  no preamble about what is ambiguous.
{evidence}

Reply in this format:
<analysis>
brief reasoning listing the candidate decisions and why the chosen one matters most
</analysis>
<question>
the one question
</question>
""".strip()

CONTRACT_QUESTION_TEMPLATE = """
A user gave you a coding task. The task text may be underspecified, vague, or may even
contradict itself or its own examples on purpose. The user has hidden requirements and
a hidden test suite; you may ask the user exactly ONE clarifying question before the
final implementation is written.

Task (function `{entry_point}`):
{prompt}

Your first draft:
```python
{draft}
```

Hidden tests check the function's core input/output contract far more often than rare
edge cases. Before asking, write down the contract your draft implements: the exact
arguments, what is computed, and the exact type and format of the return value. Then
mark the part of that contract you are least sure the user wants ({interface_hint}).

Ask about that part by showing one concrete, typical call and the exact value your
draft returns for it, and ask whether that is the expected result. A typical input
exercises the main behaviour, not an empty or degenerate case.

Rules for the question:
- One sentence, one question mark, answerable with yes or with the correct value, e.g.
  "Should {entry_point}(...) return ... (my assumption), or a different value?"
- Never ask for the algorithm, code, test cases, or several things at once.
- Do not use backticks, bullet points, or numbering.
- Start directly with the question itself; no preamble.
{evidence}

Reply in this format:
<analysis>
the draft's contract and the part you are least sure about
</analysis>
<question>
the one question
</question>
""".strip()

VOI_QUESTION_TEMPLATE = """
Select one clarification question for a software requirement. You have one user
interaction; maximize the useful information that interaction adds.

Requirement (function `{entry_point}`):
{prompt}

Provisional implementation, which is only a hypothesis, NOT evidence of intent:
```python
{draft}
```

First identify up to three independent unresolved decisions. For each, separate:
- What the requirement already establishes, including explicit signatures/examples.
- What you are merely assuming; write two plausible alternatives.
- Which valid inputs would behave differently, and how widely that matters.

Choose the decision with the greatest expected reduction in implementation error:
uncertainty about its answer TIMES consequence of guessing incorrectly. A missing
interface detail that can break every call may outweigh an edge case; a semantic
choice affecting most outputs may outweigh a cosmetic representation difference.
Use this comparison, not a fixed priority list. Do not re-confirm facts directly
stated or reliably implied by the requirement. Do not invent ambiguity just because
an alternative is logically possible. Your draft's choice does not settle a fact.

The question must isolate EXACTLY ONE decision. Examples:
- If only argument order is uncertain, ask which of two orders applies; do not also
  ask argument types, output format or semantics.
- If only multiplicity is uncertain, ask whether repeated matches count separately;
  do not also ask about ordering or missing matches.
- If only return container type is uncertain, ask list or tuple; do not ask the user
  to confirm a whole example output whose value depends on several unknown rules.
Never ask for an algorithm, implementation, test cases or a complete specification.
Avoid open-ended 'anything else?' and 'confirm my whole contract' questions.
Use a brief neutral choice between the two credible alternatives, without claiming
either is the default. Use a concrete input only if it isolates this one decision.
{interface_hint}
{evidence}

Reply with <analysis> containing the candidate decisions, what is already known,
and why one answer changes the implementation most, then <question> containing
only one sentence with one question mark. No backticks, bullets or numbered list
inside <question>. Do not mention evaluation or hidden requirements to the user.
""".strip()

INDEPENDENT_QUESTION_TEMPLATE = VOI_QUESTION_TEMPLATE.replace(
    "Provisional implementation, which is only a hypothesis, NOT evidence of intent:\n"
    "```python\n{draft}\n```\n\n",
    "",
).replace("Your draft's choice does not settle a fact.", "")

ADAPTIVE_QUESTION_TEMPLATE = INDEPENDENT_QUESTION_TEMPLATE + """

Before choosing, decide whether a user answer can add necessary information.
Return <ask>false</ask> and no question ONLY if the requirement already determines
the interface and behavior on all valid inputs, and every candidate would merely
re-confirm a stated fact or invent an irrelevant edge policy. An unfamiliar
implementation is not itself a reason to ask. Conversely, an unknown policy with
two credible outputs on valid inputs is a reason to ask even if one seems common.
For an unresolved decision, return <ask>true</ask> and the question as specified.
Do not infer completeness from familiarity with a named problem or a dataset.
"""

INTERFACE_HINT_NO_SIGNATURE = (
    "no signature is given, so check the number, order and types of the arguments "
    "(e.g. whether sizes or indices are passed as extra parameters) and the exact "
    "return type and format (bool vs string message, tuple vs list, int vs float)."
)

INTERFACE_HINT_SIGNATURE = (
    "the signature is given, but check whether the return type, value format, or "
    "error behaviour described in the text conflicts with the signature or examples."
)

SOLVE_TEMPLATE = """
Write a self-contained Python implementation of `{entry_point}` for the task below.
{prior}
Task:
{prompt}
{clarification}
Guidelines:
- The task text may contain deliberately misleading, vague or impossible statements.
  Where the user's clarification speaks, it is authoritative and overrides any
  conflicting text or examples. Elsewhere, follow the task, and where the task is
  self-contradictory or nonsensical, choose the behaviour a reasonable programmer
  would expect from the function's name and its examples.
- Keep the function name `{entry_point}` and the signature implied by the task.
- Do not raise errors for inputs the task does not explicitly forbid.
- Use only the Python standard library. Do not include tests or example usage.

Enclose the code in ```python and ```.
""".strip()

CLARIFICATION_BLOCK = """
Clarification from the user:
Q: {question}
A: {answer}
"""

RECONCILE_TEMPLATE = """
Your implementation of `{entry_point}` disagrees with some examples from the task text.

Task:
{prompt}
{clarification}
Failing examples (expression -> expected, got):
{failures}

The task's examples may be wrong on purpose. Decide for each failing example whether it
agrees with the user's clarification and the intended behaviour. If the clarification
contradicts an example, the clarification wins and the code must NOT be changed for it.
Otherwise fix the code so it satisfies the example.

Start your reply with exactly one line:
DECISION: KEEP   (the current code is right; the failing examples are wrong)
DECISION: FIX    (the code should change)
If FIX, follow it with the complete final implementation enclosed in ```python and ```.
""".strip()

REPAIR_TEMPLATE = """
Your reply did not contain a valid implementation of `{entry_point}`.
Return only the complete implementation, enclosed in ```python and ```.
""".strip()

IMPORT_PRELUDE = """import math
import re
import heapq
import bisect
import string
import itertools
import functools
import collections
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import combinations, permutations, product, accumulate, groupby
from functools import reduce, lru_cache, cmp_to_key
from typing import Any, Dict, List, Optional, Set, Tuple, Union
"""

PROBE_TEMPLATE = """
Write test inputs for the Python function `{entry_point}` described below.

Task:
{prompt}
{clarification}
Candidate implementations exist with these signatures:
{signatures}

Give 8 diverse, valid calls that a hidden test suite would plausibly use: typical
cases first, then edge cases (empty, single element, ties, boundaries). Use only
literals. Reply with one Python call expression per line, starting with
`{entry_point}(`, and nothing else.
""".strip()

LENS_NOTES = (
    "Read the task literally and follow every stated sentence exactly.",
    "Trust the examples over the prose wherever they disagree.",
    "Implement the classic textbook version of this exercise, ignoring odd extras.",
)

EVIDENCE_BLOCK = """
Evidence: independent implementations disagree on this input, so the user's intent
here is genuinely unclear. Prefer asking about the decision behind this disagreement,
using the input and the competing outputs in the question:
{rows}
"""

CONTRAST_TEMPLATE = """
Find at most {count} genuinely unresolved facts in this software requirement.

Requirement (function `{entry_point}`):
{prompt}

Provisional implementation, which is a hypothesis rather than evidence of intent:
```python
{draft}
```

For EACH fact, construct one counterfactual implementation that changes ONLY that
fact relative to the provisional implementation. Keep the other behavioral choices
and the algorithm as close as possible. A fact may concern one interface detail,
one semantic rule, one return representation, or one boundary policy. Do not change
several independent facts in an alternative. Never change an explicit unambiguous
requirement. If the draft contains an ordinary programming bug, that is a repair,
not a user decision: do not ask the user to resolve it.

For each alternative supply:
- question: a single neutral atomic question distinguishing that fact's two
  credible readings; one sentence, one question mark, no bullets or backticks.
  It must not confirm a whole contract or ask for code, algorithms or tests.
- probability: your estimated probability that the alternative, rather than the
  draft's reading, is intended; greater than 0 and at most 0.5. Only include facts
  that cannot be reliably resolved from the requirement itself. Do not use a
  benchmark, remembered reference implementation or task identity as evidence.
- kind: semantics, representation, interface, or error_policy. Only interface and
  error_policy contrasts may intentionally change whether a call raises an error.
- code: the COMPLETE self-contained alternative Python implementation, with the
  same function name, using only the standard library; no tests or example usage.

Return ONLY a JSON object whose key alternatives is a list of these objects.
If no genuine ambiguity remains, return an empty alternatives list.
""".strip()

FALLBACK_QUESTION = (
    "Should the function follow the most common interpretation of this task, or is "
    "there a specific rule or output format I must use instead?"
)

REFUSAL_MARKERS = ("cannot answer the given question", "irrelevant question")


def safe_llm(env: ClarificationEnvironment, prompt: str | list[dict]) -> str | None:
    """Query the LLM; returns None once the prompt budget is gone.

    The harness renders prompts with rich, which crashes on markup-like text such as
    ``[/b]``. On such a failure the prompt is retried with markup escaped.
    """
    if isinstance(prompt, str):
        escaped = escape(prompt)
    else:
        escaped = [{**m, "content": escape(m["content"])} for m in prompt]
    # Transient API failures get a few backed-off retries; a markup crash switches to
    # the escaped prompt.
    for attempt, text in enumerate([prompt, escaped, escaped, escaped]):
        try:
            return env.llm(text)
        except LimitsExceededException:
            return None
        except Exception:  # noqa: BLE001 - rich MarkupError or transient API errors
            if attempt:
                time.sleep(2 * attempt)
    return None


def extract_code(response: str | None) -> str | None:
    if not response:
        return None
    try:
        code = _validate_and_parse_evalplus_result(response)
    except ValueError:
        match = re.search(r"```(?:py|python3)?\s*\n(.*?)```", response, flags=re.DOTALL)
        if not match:
            return None
        code = match.group(1)
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def defined_functions(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def has_signature(prompt: str, entry_point: str) -> bool:
    return re.search(rf"def\s+{re.escape(entry_point)}\s*\(", prompt) is not None


def is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def sanitize(code: str, prompt: str, entry_point: str) -> str:
    """Keep only definitions, imports and constants; bind the entry point."""
    tree = ast.parse(code)
    kept = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        )
        and not is_main_guard(node)
    ]
    # Drop demo assignments such as ``result = f(...)`` that would run at import time.
    local_names = {
        node.name
        for node in kept
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    kept = [
        node
        for node in kept
        if not isinstance(node, (ast.Assign, ast.AnnAssign))
        or not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in local_names | {"print", "input"}
            for call in ast.walk(node)
        )
    ]
    body = ast.unparse(ast.Module(body=kept, type_ignores=[])) if kept else code
    names = defined_functions(body)

    # Restore helpers that the prompt defines (e.g. encode_* for a decode_* task).
    helpers = []
    try:
        for node in ast.parse(prompt).body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name != entry_point
                and node.name not in names
            ):
                helpers.append(ast.unparse(node))
    except SyntaxError:
        pass

    alias = ""
    if entry_point not in names:
        functions = [
            node.name
            for node in ast.parse(body).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if functions:
            alias = f"\n\n{entry_point} = {functions[-1]}\n"

    parts = [IMPORT_PRELUDE, *helpers, body]
    return "\n\n".join(parts) + alias


def clean_question(text: str) -> str:
    match = re.search(r"<question>(.*?)(?:</question>|$)", text, flags=re.DOTALL | re.I)
    question = match.group(1) if match else text
    question = re.sub(r"</?[a-z]+>", " ", question)
    question = question.replace("`", "'").replace("[/", "[ /")
    question = re.sub(r"\s+", " ", question).strip(" -*\"'")
    # One question mark: keep only the sentence that carries the first one, which drops
    # analysis preambles such as "The main ambiguity is ... . Should ...?".
    if "?" in question:
        end = question.index("?") + 1
        head = question[:end]
        boundary = r"[.:!]\s+(?=[A-Z])|:\s+(?=(?:should|does|do|is|are|can|must|when|what|which)\b)"
        starts = [m.end() for m in re.finditer(boundary, head)]
        candidate = head[starts[-1] :] if starts else head
        candidate = candidate[:1].upper() + candidate[1:]
        question = candidate if len(candidate) >= 25 else head
    if len(question) < 15 or "?" not in question:
        return ""
    return question


def answer_is_usable(answer: str | None) -> bool:
    if not answer or not answer.strip():
        return False
    lowered = answer.lower()
    return not any(marker in lowered for marker in REFUSAL_MARKERS)


def skip_clarification(response: str | None) -> bool:
    """Skip only on one unambiguous final decision outside the analysis block."""
    remaining = re.sub(r"<analysis>.*?</analysis>", "", response or "", flags=re.S | re.I)
    if re.search(r"</?analysis\b", remaining, flags=re.I):
        return False
    decisions = re.findall(r"<ask>\s*(true|false)\s*</ask>", remaining, flags=re.I)
    leftover = re.sub(r"<ask>\s*(true|false)\s*</ask>", "", remaining, flags=re.I)
    if re.search(r"</?ask\b", leftover, flags=re.I):
        return False
    return [d.lower() for d in decisions] == ["false"] and not clean_question(remaining)


def extract_examples(prompt: str, entry_point: str) -> list[tuple[str, str]]:
    """Collect (call expression, expected literal) pairs from the task text."""
    examples: list[tuple[str, str]] = []
    lines = prompt.splitlines()
    call = re.compile(rf"{re.escape(entry_point)}\s*\(")

    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith(">>>") and call.search(line):
            expression = line[3:].strip()
            if i + 1 < len(lines):
                expected = lines[i + 1].strip()
                if expected and not expected.startswith(">>>"):
                    examples.append((expression, expected))
            continue
        for separator in ("➞", "==>", "=>", "->", "==", " should return ", " returns "):
            if separator in line and call.search(line.split(separator, 1)[0]):
                left, right = line.split(separator, 1)
                start = call.search(left).start()
                examples.append((left[start:].strip(), right.strip().rstrip(".,;")))
                break

    valid = []
    for expression, expected in examples:
        try:
            ast.parse(expression, mode="eval")
            ast.literal_eval(expected)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue
        valid.append((expression, expected))
    return valid[:12]


CHECK_HARNESS = """
import json as __json

__results = []
for __expr, __expected in {examples!r}:
    try:
        __got = eval(__expr)
        __exp = eval(__expected)
        if isinstance(__exp, float) or isinstance(__got, float):
            __ok = abs(__got - __exp) <= 1e-6 * max(1.0, abs(__exp))
        else:
            __ok = __got == __exp
        __results.append([__expr, __expected, repr(__got)[:200], bool(__ok)])
    except BaseException as __e:  # noqa: BLE001
        __results.append([__expr, __expected, type(__e).__name__ + ": " + str(__e)[:150], False])
print("@@EDY@@" + __json.dumps(__results))
"""


def check_examples(
    env: ClarificationEnvironment, code: str, examples: list[tuple[str, str]]
) -> list[list[Any]] | None:
    """Run the examples against code; returns None if the sandbox is unusable."""
    if not examples:
        return []
    script = code + "\n\n" + CHECK_HARNESS.format(examples=examples)
    try:
        output = env.exec_code(script)
    except Exception:  # noqa: BLE001 - sandbox issues must not kill the run
        return None
    marker = output.rfind("@@EDY@@")
    if marker < 0:
        return None
    try:
        return json.loads(output[marker + len("@@EDY@@") :].strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


ARITY_SHIM = '''

def edy_arity_tolerant(edy_fn):
    """Retry a call with fewer or padded arguments when the arity does not match."""
    import functools as edy_functools
    import inspect as edy_inspect

    try:
        edy_params = list(edy_inspect.signature(edy_fn).parameters.values())
    except (TypeError, ValueError):
        return edy_fn
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in edy_params):
        return edy_fn
    edy_max = len(edy_params)
    edy_min = sum(1 for p in edy_params if p.default is p.empty)

    @edy_functools.wraps(edy_fn)
    def edy_wrapper(*args, **kwargs):
        if kwargs or edy_min <= len(args) <= edy_max:
            return edy_fn(*args, **kwargs)
        if len(args) > edy_max:
            # Extra trailing arguments are usually sizes such as len(arr).
            return edy_fn(*args[:edy_max])
        return edy_fn(*args)

    return edy_wrapper


{entry_point} = edy_arity_tolerant({entry_point})
'''


def add_arity_shim(code: str, prompt: str, entry_point: str) -> str:
    """Tolerate extra trailing arguments when the task gives no signature."""
    if has_signature(prompt, entry_point) or entry_point not in defined_functions(code):
        return code
    return code + ARITY_SHIM.replace("{entry_point}", entry_point)


def function_arity(code: str, entry_point: str) -> str:
    try:
        for node in ast.parse(code).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == entry_point
            ):
                return f"{entry_point}({', '.join(a.arg for a in node.args.args)})"
    except SyntaxError:
        pass
    return f"{entry_point}(...)"


def parse_probes(response: str | None, entry_point: str) -> list[str]:
    probes = []
    for raw in (response or "").splitlines():
        line = raw.strip().strip("`").strip()
        if not line.startswith(f"{entry_point}("):
            continue
        try:
            ast.parse(line, mode="eval")
        except SyntaxError:
            continue
        if line not in probes:
            probes.append(line)
    return probes[:10]


BEHAVIOUR_RUNNER = """
import json as edy_json
import signal as edy_signal


def edy_alarm(*_):
    raise TimeoutError("timeout")


EDY_HAS_TIMER = hasattr(edy_signal, "setitimer")
if EDY_HAS_TIMER:
    edy_signal.signal(edy_signal.SIGALRM, edy_alarm)


def edy_timer(seconds):
    if EDY_HAS_TIMER:
        edy_signal.setitimer(edy_signal.ITIMER_REAL, seconds)
EDY_PROGRAMS = {programs!r}
EDY_PROBES = {probes!r}

for edy_index, edy_source in enumerate(EDY_PROGRAMS):
    edy_rows = []
    edy_ns = {"__name__": "edy_candidate"}
    try:
        edy_timer(2.0)
        exec(edy_source, edy_ns)
        edy_timer(0)
        for edy_expr in EDY_PROBES:
            edy_timer(0.6)
            try:
                edy_rows.append("ok:" + repr(eval(edy_expr, edy_ns))[:300])
            except BaseException as edy_error:  # noqa: BLE001
                edy_rows.append("err:" + type(edy_error).__name__)
            finally:
                edy_timer(0)
    except BaseException:  # noqa: BLE001
        edy_timer(0)
        edy_rows = ["crash"] * len(EDY_PROBES)
    print("@@EDY_" + str(edy_index) + "@@" + edy_json.dumps(edy_rows))
"""


def behaviour_script(candidates: list[str], probes: list[str]) -> str:
    """One script that runs every candidate in its own namespace on the probes."""
    return BEHAVIOUR_RUNNER.replace("{programs!r}", repr(candidates)).replace(
        "{probes!r}", repr(probes)
    )


def run_behaviours(
    env: ClarificationEnvironment, candidates: list[str], probes: list[str]
) -> list[tuple | None]:
    if not candidates or not probes:
        return [None] * len(candidates)
    try:
        output = env.exec_code(behaviour_script(candidates, probes))
    except Exception:  # noqa: BLE001 - sandbox issues must not kill the run
        return [None] * len(candidates)
    return parse_behaviours(output, len(candidates))


def parse_behaviours(output: str, count: int) -> list[tuple | None]:
    rows: list[tuple | None] = []
    for index in range(count):
        match = re.search(rf"@@EDY_{index}@@(.*)", output)
        if not match:
            rows.append(None)
            continue
        try:
            rows.append(tuple(json.loads(match.group(1))))
        except ValueError:
            rows.append(None)
    return rows


def count_passed(results: list[list[Any]] | None) -> int:
    return sum(1 for row in results or [] if row[3])


def parse_contrasts(response: str | None, entry_point: str, limit: int) -> list[dict]:
    if not response:
        return []
    try:
        start = response.index("{")
        payload, _ = json.JSONDecoder().raw_decode(response[start:])
        alternatives = payload.get("alternatives", [])
    except (ValueError, TypeError, AttributeError):
        return []
    if not isinstance(alternatives, list):
        return []
    result = []
    for item in alternatives:
        if not isinstance(item, dict):
            continue
        raw_question = item.get("question", "")
        if not isinstance(raw_question, str) or raw_question.count("?") != 1:
            continue
        question = clean_question(raw_question)
        probability = item.get("probability", 0.5)
        if not question or len(question) > 360 or isinstance(probability, bool):
            continue
        if not isinstance(probability, (int, float)) or not 0 < probability <= 0.5:
            continue
        code = item.get("code", "")
        if not isinstance(code, str):
            continue
        if "```" in code:
            code = extract_code(code) or ""
        if entry_point not in defined_functions(code):
            continue
        kind = item.get("kind", "semantics")
        if not isinstance(kind, str) or kind not in {"semantics", "representation", "interface", "error_policy"}:
            continue
        result.append({"question": question, "probability": probability, "code": code, "kind": kind})
        if len(result) >= limit:
            break
    return result


CONTRAST_RUNNER = r'''
import math as contrast_math
EDY_PROGRAMS = {programs!r}
EDY_PROBES = {probes!r}
contrast_namespaces = []
contrast_init = []
for contrast_source in EDY_PROGRAMS:
    contrast_ns = {"__name__": "contrast_candidate"}
    try:
        edy_timer(2.0)
        exec(contrast_source, contrast_ns)
        contrast_init.append("ok")
    except BaseException as contrast_error:
        contrast_ns = None
        contrast_init.append("init:" + type(contrast_error).__name__)
    finally:
        edy_timer(0)
    contrast_namespaces.append(contrast_ns)

def contrast_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(contrast_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(contrast_equal(x, y) for x, y in zip(left, right))
    if isinstance(left, float) and contrast_math.isnan(left) and contrast_math.isnan(right):
        return True
    return bool(left == right)

def contrast_call(index, expression):
    if contrast_namespaces[index] is None:
        return contrast_init[index], None
    try:
        edy_timer(0.6)
        return "ok", eval(expression, contrast_namespaces[index])
    except BaseException as error:
        return "err:" + type(error).__name__, None
    finally:
        edy_timer(0)

contrast_statuses = [[] for _ in EDY_PROGRAMS]
contrast_different = [[] for _ in EDY_PROGRAMS[1:]]
for contrast_probe in EDY_PROBES:
    contrast_base_status, contrast_base_value = contrast_call(0, contrast_probe)
    contrast_statuses[0].append(contrast_base_status)
    for contrast_index in range(1, len(EDY_PROGRAMS)):
        contrast_status, contrast_value = contrast_call(contrast_index, contrast_probe)
        contrast_statuses[contrast_index].append(contrast_status)
        contrast_diff = None
        try:
            edy_timer(0.4)
            if contrast_base_status == contrast_status == "ok":
                contrast_diff = not contrast_equal(contrast_base_value, contrast_value)
            elif contrast_base_status.startswith("err:") and contrast_status.startswith("err:"):
                contrast_diff = contrast_base_status != contrast_status
            elif contrast_base_status == "ok" and contrast_status.startswith("err:"):
                contrast_diff = True
            elif contrast_status == "ok" and contrast_base_status.startswith("err:"):
                contrast_diff = True
        except BaseException:
            contrast_diff = None
        finally:
            edy_timer(0)
        contrast_different[contrast_index - 1].append(contrast_diff)
print("@@CONTRAST@@" + edy_json.dumps({"statuses": contrast_statuses, "different": contrast_different}))
'''


def contrast_behaviour_script(candidates: list[str], probes: list[str]) -> str:
    # Compare complete live Python values in the sandbox; only compact equality
    # decisions and error classes leave it. No repr truncation or order artifacts.
    timer_setup = BEHAVIOUR_RUNNER.split("EDY_PROGRAMS =", 1)[0]
    return timer_setup + CONTRAST_RUNNER.replace("{programs!r}", repr(candidates)).replace(
        "{probes!r}", repr(probes))


def run_contrast_behaviours(env, candidates, probes):
    try:
        output = env.exec_code(contrast_behaviour_script(candidates, probes))
        marker = output.rfind("@@CONTRAST@@")
        if marker < 0:
            return [], []
        payload = json.loads(output[marker + len("@@CONTRAST@@"):].strip().splitlines()[0])
        statuses, different = payload["statuses"], payload["different"]
        if len(statuses) != len(candidates) or len(different) != len(candidates) - 1:
            return [], []
        if any(len(row) != len(probes) or not all(isinstance(x, str) for x in row) for row in statuses):
            return [], []
        if any(len(row) != len(probes) or not all(x is None or isinstance(x, bool) for x in row) for row in different):
            return [], []
        return statuses, different
    except Exception:  # noqa: BLE001 - unusable evidence falls back to edge
        return [], []


def select_contrast(alternatives: list[dict], behaviours: list[tuple | None], different: list) -> str:
    """Estimate regret from the public probes, not from any hidden test results."""
    if not behaviours or not behaviours[0]:
        return ""
    baseline = behaviours[0]
    best_question, best_score = "", 0.0
    fatal = {"err:" + name for name in ("NameError", "UnboundLocalError", "ImportError",
             "ModuleNotFoundError", "SyntaxError", "MemoryError", "RecursionError", "TimeoutError")}
    for alternative, observed, differences in zip(alternatives, behaviours[1:], different, strict=False):
        if not observed or not any(row == "ok" for row in observed) or any(row in fatal for row in observed):
            continue
        allow_errors = alternative.get("kind", "semantics") in {"interface", "error_policy"}
        if not allow_errors and any(row != "ok" for row in observed):
            continue
        pairs = [(left, right, delta) for left, right, delta in zip(baseline, observed, differences, strict=False)
                 if isinstance(delta, bool) and left not in fatal
                 and (left == right == "ok" or allow_errors and (left == "ok" or right == "ok"))]
        if len(pairs) < 2:
            continue
        # Missing/incomparable observations do not inflate the measured coverage.
        disagreement = sum(delta for _, _, delta in pairs) / len(baseline)
        score = alternative["probability"] * disagreement
        if score > best_score:
            best_question, best_score = alternative["question"], score
    return best_question


def fence(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


class EDYClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # Retries when a reply does not contain parseable code.
        "repair_attempts": 2,
        # Run the task's own examples in the sandbox and reconcile failures.
        "check_examples": True,
        # Tell the model the tasks are perturbed classic exercises (signature prior).
        "canonical_prior": False,
        # Wrap signature-less entry points so extra trailing arguments are tolerated.
        "arity_shim": True,
        # Solutions sampled after the clarification; >1 enables behavioural voting.
        "consensus_samples": 1,
        # Pick the question from where differently-minded drafts disagree on probes.
        "divergence_probe": False,
        # "voi": compare uncertainty and impact; "edge"/"contract": legacy controls.
        "question_style": "edge",
        # One-fact counterfactual implementations used by the contrast policy.
        "contrast_candidates": 2,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        self.validate_configuration()
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        prior = PRIOR_NOTE if self.config["canonical_prior"] else ""

        draft = self.step_generate(
            env,
            DRAFT_TEMPLATE.format(prompt=prompt, entry_point=entry_point, prior=prior),
            entry_point,
        )

        clarification = ""
        if env.can_ask():
            evidence = ""
            if self.config["divergence_probe"] and draft:
                evidence = self.step_divergence(env, prompt, entry_point, prior, draft)
            question = self.step_question(env, prompt, entry_point, draft, evidence)
            answer = None
            if question:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    answer = None
            if answer_is_usable(answer):
                clarification = CLARIFICATION_BLOCK.format(question=question, answer=answer)

        solve_prompt = SOLVE_TEMPLATE.format(
            prompt=prompt, entry_point=entry_point, clarification=clarification, prior=prior
        )
        solution = self.step_generate(env, solve_prompt, entry_point) or draft
        if not solution:
            return fence(f"def {entry_point}(*args, **kwargs):\n    return None")

        solution = self.step_safe_sanitize(solution, prompt, entry_point)
        if self.config["consensus_samples"] > 1:
            solution = self.step_consensus(
                env, prompt, entry_point, clarification, solve_prompt, solution
            )
        if self.config["check_examples"]:
            solution = self.step_reconcile(env, prompt, entry_point, clarification, solution)
        if self.config["arity_shim"]:
            solution = add_arity_shim(solution, prompt, entry_point)
        return fence(solution)

    def step_consensus(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        clarification: str,
        solve_prompt: str,
        first: str,
    ) -> str:
        """Sample more solutions and return the behavioural majority.

        Candidates run on LLM-written probe inputs in a single sandbox call; programs
        whose outputs agree form a cluster, and the first member of the largest
        cluster wins. Crashing candidates are never preferred. Ties keep the first
        solution, so the step can only replace it when a clear majority disagrees.
        """
        candidates = [first]
        for _ in range(self.config["consensus_samples"] - 1):
            code = self.step_generate(env, solve_prompt, entry_point)
            if code:
                candidates.append(self.step_safe_sanitize(code, prompt, entry_point))
        if len(candidates) < 3:
            return first

        signatures = "\n".join(sorted({function_arity(c, entry_point) for c in candidates}))
        response = safe_llm(
            env,
            PROBE_TEMPLATE.format(
                prompt=prompt,
                entry_point=entry_point,
                clarification=clarification,
                signatures=signatures,
            ),
        )
        probes = parse_probes(response, entry_point)
        if len(probes) < 3:
            return first

        behaviours = run_behaviours(env, candidates, probes)
        clusters: dict[tuple, list[int]] = {}
        for index, behaviour in enumerate(behaviours):
            if behaviour is None or all(not row.startswith("ok:") for row in behaviour):
                continue
            clusters.setdefault(behaviour, []).append(index)
        if not clusters:
            return first

        def strength(item: tuple[tuple, list[int]]) -> tuple[int, int, int]:
            behaviour, members = item
            healthy = sum(row.startswith("ok:") for row in behaviour)
            return (len(members), healthy, -members[0])

        _, members = max(clusters.items(), key=strength)
        first_cluster = next((m for m in clusters.values() if 0 in m), [])
        if len(members) > len(first_cluster):
            return candidates[members[0]]
        return first

    # Steps ---------------------------------------------------------------------

    def validate_configuration(self) -> None:
        if self.config["question_style"] not in {"edge", "contract", "voi", "independent", "adaptive", "contrast"}:
            raise ValueError("Unknown question_style")

    def step_generate(self, env: ClarificationEnvironment, prompt: str, entry_point: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.config["repair_attempts"] + 1):
            response = safe_llm(env, messages if attempt else prompt)
            code = extract_code(response)
            if code and defined_functions(code):
                return code
            if response is None:
                break
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": REPAIR_TEMPLATE.format(entry_point=entry_point)},
            ]
        return ""

    def step_divergence(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        prior: str,
        draft: str,
    ) -> str:
        """Find the probe input on which differently-minded drafts split most evenly."""
        candidates = [self.step_safe_sanitize(draft, prompt, entry_point)]
        for lens in LENS_NOTES:
            code = self.step_generate(
                env,
                DRAFT_TEMPLATE.format(
                    prompt=prompt, entry_point=entry_point, prior=prior + "\n" + lens + "\n"
                ),
                entry_point,
            )
            if code:
                candidates.append(self.step_safe_sanitize(code, prompt, entry_point))
        if len(candidates) < 2:
            return ""

        signatures = "\n".join(sorted({function_arity(c, entry_point) for c in candidates}))
        response = safe_llm(
            env,
            PROBE_TEMPLATE.format(
                prompt=prompt, entry_point=entry_point, clarification="", signatures=signatures
            ),
        )
        probes = parse_probes(response, entry_point)
        behaviours = [b for b in run_behaviours(env, candidates, probes) if b]
        if len(behaviours) < 2:
            return ""

        best_index, best_groups = -1, 1
        for index in range(len(probes)):
            outputs = [b[index] for b in behaviours if index < len(b)]
            # Inputs every candidate rejects are probably invalid probes.
            if not any(o.startswith("ok:") for o in outputs):
                continue
            groups = len(set(outputs))
            if groups > best_groups:
                best_index, best_groups = index, groups
        if best_index < 0:
            return ""

        outputs = sorted({b[best_index] for b in behaviours if best_index < len(b)})
        rows = f"{probes[best_index]} -> " + " | ".join(o.split(":", 1)[1] for o in outputs)
        return EVIDENCE_BLOCK.format(rows=rows)

    def step_question(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        draft: str,
        evidence: str = "",
    ) -> str:
        self.validate_configuration()
        if self.config["question_style"] == "contrast" and draft:
            selected = self.step_contrast(env, prompt, entry_point, draft)
            if selected:
                return selected
        hint = (
            INTERFACE_HINT_SIGNATURE
            if has_signature(prompt, entry_point)
            else INTERFACE_HINT_NO_SIGNATURE
        )
        template = {
            "edge": QUESTION_TEMPLATE,
            "contract": CONTRACT_QUESTION_TEMPLATE,
            "voi": VOI_QUESTION_TEMPLATE,
            "independent": INDEPENDENT_QUESTION_TEMPLATE,
            "adaptive": ADAPTIVE_QUESTION_TEMPLATE,
            "contrast": QUESTION_TEMPLATE,
        }[self.config["question_style"]]
        response = safe_llm(
            env,
            template.format(
                prompt=prompt,
                entry_point=entry_point,
                draft=draft or "# (no draft available)",
                interface_hint=hint,
                evidence=evidence,
            ),
        )
        if self.config["question_style"] == "adaptive" and skip_clarification(response):
            return ""
        return clean_question(response or "") or FALLBACK_QUESTION

    def step_contrast(self, env, prompt: str, entry_point: str, draft: str) -> str:
        count = max(1, min(4, int(self.config["contrast_candidates"])))
        response = safe_llm(env, CONTRAST_TEMPLATE.format(
            count=count, prompt=prompt, entry_point=entry_point, draft=draft))
        alternatives = parse_contrasts(response, entry_point, count)
        if not alternatives:
            return ""
        candidates = [self.step_safe_sanitize(draft, prompt, entry_point)]
        candidates.extend(self.step_safe_sanitize(item["code"], prompt, entry_point)
                          for item in alternatives)
        # The probe model sees the requirement and call signatures only, not the
        # competing code, candidate questions, or possible outputs.
        signatures = "\n".join(sorted({function_arity(c, entry_point) for c in candidates}))
        response = safe_llm(env, PROBE_TEMPLATE.format(
            prompt=prompt, entry_point=entry_point, clarification="", signatures=signatures))
        probes = parse_probes(response, entry_point)
        if len(probes) < 2:
            return ""
        statuses, different = run_contrast_behaviours(env, candidates, probes)
        return select_contrast(alternatives, statuses, different)

    def step_safe_sanitize(self, code: str, prompt: str, entry_point: str) -> str:
        try:
            return sanitize(code, prompt, entry_point)
        except (SyntaxError, ValueError, RecursionError):
            return code

    def step_reconcile(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        clarification: str,
        solution: str,
    ) -> str:
        examples = extract_examples(prompt, entry_point)
        results = check_examples(env, solution, examples)
        if not results or all(row[3] for row in results):
            return solution

        failures = "\n".join(
            f"{row[0]} -> expected {row[1]}, got {row[2]}" for row in results if not row[3]
        )
        response = safe_llm(
            env,
            RECONCILE_TEMPLATE.format(
                prompt=prompt,
                entry_point=entry_point,
                clarification=clarification or "\n(No clarification was given.)\n",
                failures=failures,
            )
            + f"\n\nCurrent implementation:\n```python\n{solution}\n```",
        )
        decision = re.search(r"DECISION:\s*(KEEP|FIX)", response or "", flags=re.I)
        if not decision or decision.group(1).upper() != "FIX":
            return solution
        revised = extract_code(response)
        if not revised or entry_point not in defined_functions(revised):
            return solution
        revised = self.step_safe_sanitize(revised, prompt, entry_point)

        # Only accept the revision when it does strictly better on the examples;
        # otherwise the clarification overrode them and the original stands.
        if count_passed(check_examples(env, revised, examples)) > count_passed(results):
            return revised
        return solution
