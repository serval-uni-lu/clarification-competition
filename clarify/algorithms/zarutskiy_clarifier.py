# SPDX-FileCopyrightText: 2026 Semyon Zarutskiy <zarutskiysy@gmail.com>
#
# SPDX-License-Identifier: MIT

"""ZarutskiyClarifier.

Always asks exactly one clarifying question, which reads the seed draft's contract back to
the user and ends with an open invitation to say what should be different, then writes one
final version from the task plus the question and answer, with the answer overriding
conflicting task text.

One LLM call states, from the task and the draft, what the draft expects as input (in order,
with types), what it gives back, the rule it assumes at the point the description leaves
most open or contradicts, and one call on a tiny valid input that shows that rule. One
env.exec_code call runs the draft on that call, and the question reads "I will assume f
expects ... and gives back ..., and that .... So f(...) would return X. Is that right, or
did you mean something else?", where X is the draft's real output; the example is left out
when the call raises or its output is long. This is done on every task, with or without a
signature. If a field is missing, the question of the previous version (ae88217) is asked
instead: one concrete, atomic question about the single missing, vague or contradictory
requirement that changes the output, phrased "I will assume X. Is that right, or should it
be Y?", where X is what the draft does.

A deterministic sanitizer aliases the required entry point when a draft renamed the function
and drops __main__ blocks and top-level prints; questions are sent without backticks,
because the simulated user's answer parser stops at the first backtick. If anything fails
after the seed draft, the sanitized seed draft is returned. env.llm calls are retried up to
twice on errors other than the budget limit. A task uses 3 env.llm calls (4 when the
read-back falls back), at most one env.exec_code call and one env.ask_human call.
DEFAULT_CONFIG also holds switches from earlier experiments (other question styles and
framings, disagreement probing, execution-based selection); with these defaults they are off
or not reached.

Credits. The contract read-back with an executed example follows ContrastiveClarifier (Dinh,
PR #9), InfoGainClarifier (RankOne, PR #12; also the read-back on every task, with or
without a signature) and the contract question of CrossCheckClarification (STIL-ETS, PR #4);
the open close follows ContrastiveClarifier and InfoGainClarifier. The base is our own P05
assumption framing (ae88217): the read-back extends its assumption from one point to the
draft's whole contract, and the P05 question is the fallback. The code and the new prompts
are our own; the code-generation prompt is the Okanagan baseline's (Wu and Fard, TOSEM 2025)
with our rules added, and the okanagan options (off by default) use its question and
regeneration prompts. Earlier ideas still in use: fresh regeneration with the answer taking
precedence follows GatedClarification (STIL-ETS, the earlier version of PR #4); the
disagreement option (off by default) follows ClarifyGPT (Mu et al., FSE 2024); AST hygiene
follows ContractFirstClarifier (D4vidHuang) and GatedClarification; the backtick gotcha
follows the CuriosityByDesign notes (thibolu-uofa).

Team: zarutskiysy
Team Members: Semyon Zarutskiy, Veronica Veselova
Main Contact: zarutskiysy@gmail.com
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

CODE_PROMPT = """
Generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`.

{prompt}

{name_rule}Enclose your solution in ```python and ```.
""".strip()

# Okanagan's question prompt (baseline wording), used for question_style="okanagan".
OKANAGAN_QUESTION_PROMPT = """
Given the programming problem and a generated candidate, ask ONE clarifying question if the requirements in the given problem description are incomplete, inconsistent or ambiguous
for solving the problem correctly and passing the tests.
If no need to ask clarifying questions, return strictly ’NO_QUESTIONS’ only. Otherwise, return the clarifying questions.

### Problem:

{prompt}

### Candidate

{candidate}
""".strip()

# Same wording without the opt-out, used for ask_policy="always" with question_style="okanagan".
OKANAGAN_ALWAYS_PROMPT = """
Given the programming problem and a generated candidate, ask ONE clarifying question about the requirement in the given problem description that is most likely incomplete, inconsistent or ambiguous
for solving the problem correctly and passing the tests.
Return only the clarifying question.

### Problem:

{prompt}

### Candidate

{candidate}
""".strip()

TARGETED_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

Your first draft:
```python
{candidate}
```

Before you finalize the code you may ask the user exactly ONE clarifying question; the user knows the intended behavior, and there is no second chance.

Find the single point where the description is missing information, is vague, or contradicts itself (text vs. examples, text vs. signature, two statements that cannot both hold) in a way that changes what `{entry_point}` must return or accept for ordinary inputs. Prefer the point where your draft is most likely to be wrong.

Answer in exactly this format:
ANALYSIS: <2-4 sentences: the competing readings and how their outputs differ>
QUESTION: <one short question about exactly one fact, answerable with a single value or choice. When it helps, name the concrete alternatives or a tiny example input, e.g. "Should f([3, 1, 3]) return 2 or 3?">

{interface_hint}Rules for the question: ask about required behavior (a value, a rule, a return type, an input format, an edge-case convention), never about implementation details or test cases; do not bundle several questions; do not ask to confirm something the description already states clearly. Write it in plain text: no Markdown, no backticks.
""".strip()

INTERFACE_QUESTION = "Which arguments does {entry_point} take, and in what order?"

ASSUME_FRAMING_TEXT = 'QUESTION: <one short question about exactly one fact, phrased as an assumption for the user to confirm or correct: "I will assume X. Is that right, or should it be Y?", where X is what your draft does at that point and Y is the most plausible other reading. Name concrete values or a tiny example input when it helps, e.g. "I will assume f([3, 1, 3]) returns 2. Is that right, or should it return 3?">'

ASSUME_NOY_TEXT = 'QUESTION: <one short question about exactly one fact, phrased as an assumption for the user to confirm or correct: "I will assume X. Is that right?", where X is what your draft does at that point. Name concrete values or a tiny example input when it helps, e.g. "I will assume f([3, 1, 3]) returns 2. Is that right?">'

# XE: the assumption is stated as what the draft really returns on one call (filled in by execution).
EXEC_ASSUME_TEXT = """CALL: <one call of `{entry_point}` on a tiny, valid input where the competing readings give different results, written as a single Python expression, e.g. f([3, 1, 3])>
QUESTION: <one short question about exactly one fact, phrased as an assumption for the user to confirm or correct, of the form "I will assume <the call> returns RESULT. Is that right?". Write the word RESULT literally: it will be replaced by what your draft actually returns for that call. If no single call can show the point, ask "I will assume X. Is that right?" in words instead.>"""

# IFA: on prompts without a signature, the assumption is about how the function is called.
IFACE_ASSUME_TEXT = 'QUESTION: <one short question about the call interface, phrased as an assumption for the user to confirm or correct: "I will assume {entry_point} is called like C. Is that right?", where C is one concrete example call with the number, order and types of arguments your draft expects, e.g. "I will assume f is called like f([3, 1, 3], 2). Is that right?">'

# CR: P05's assumption extended from one point to the draft's whole contract, with the example output
# taken from executing the draft. The read-back idea comes from ContrastiveClarifier (Dinh), InfoGainClarifier
# (RankOne) and CrossCheckClarification (STIL-ETS); the wording is ours.
READBACK_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

Your first draft:
```python
{candidate}
```

Before you finalize the code you will state your assumptions about the task in one message, and the user will confirm or correct them; there is no second chance. State what your draft assumes, in plain words a user can check.

Answer in exactly this format, one line per field, plain text without Markdown or backticks:
ARGS: <what your draft expects as input, in order, with types, e.g. a list of integers and an integer k>
OUTPUT: <what it gives back, with type and format, e.g. the number of pairs as an int>
ASSUMPTION: <one short clause: your draft's reading of the point the description leaves most open or contradicts, e.g. pairs are counted with i < j and equal values count>
CALL: <one call of {entry_point} on a tiny, valid input that shows that assumption, as a single Python expression, e.g. f([3, 1, 3], 2)>
""".strip()

READBACK_TEMPLATE = "I will assume {entry_point} expects {args} and gives back {output}, and that {assumption}.{example} {close}"
READBACK_CLOSE = "Is that right?"
READBACK_CLOSE_OPEN = "Is that right, or did you mean something else?"

TESTS_SEL_PROMPT = """
Task for `{entry_point}`:

<problem>
{prompt}
</problem>
{clarification}
Write up to {num_tests} assert statements that check `{entry_point}` on concrete, valid inputs. Use the examples stated in the task (where they agree with the clarification) and at least one assert that encodes the user's answer, when the answer states a concrete behavior. The expected value must follow unambiguously from the task text, its examples or the answer; skip a case rather than guess.
Return only a Python code block with one `assert {entry_point}(...) == ...` statement per line.
""".strip()

REPAIR_PROMPT = """
Implement `{entry_point}` for the task below.

<problem>
{prompt}
</problem>
{clarification}
Your current implementation:
```python
{code}
```

These checks were written from the task text, its examples and the user's answer, and your implementation fails them:
{failures}

If a check follows from the task or the answer, fix the implementation so it passes; if a check contradicts the task or the answer, ignore that check. Keep the function name `{entry_point}` and everything that already works.
Enclose your solution in ```python and ```.
""".strip()

INTERFACE_HINT = """The description gives no signature, so the user's tests will call `{entry_point}` with arguments whose number, order and types you have to guess. If your draft's parameters are not clearly determined by the description, ask about them (e.g. "Does the function take the list and the target value as two separate arguments, in that order?").

"""

MULTI_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

Your first draft:
```python
{candidate}
```

Before you finalize the code you may send the user ONE message with clarifying questions; there is no second chance.
List the points where the description is missing information, is vague, or contradicts itself in a way that changes what `{entry_point}` must return for ordinary inputs, most important first.

Answer in exactly this format:
ANALYSIS: <2-4 sentences>
QUESTIONS:
1. <short question about one fact>
2. <short question about one fact>
3. <short question about one fact>

Ask at most {max_questions} questions, each about required behavior (a value, a rule, a return type, an input format, an edge-case convention), never about implementation details or test cases. Write them in plain text: no Markdown, no backticks.
""".strip()

PROBE_PROMPT = """
Problem for `{entry_point}`:

{prompt}

Propose {num_probes} short argument lists for `{entry_point}` that could separate different reasonable interpretations of this description: the examples it gives (if any), typical inputs, and edge inputs (empty, ties, duplicates, zero, limits) where interpretations might differ. Only use inputs the description permits; leave out types or values the function would reject.
Answer with a single Python code block assigning INPUTS, a list in which each item is the tuple of positional arguments for one call, like:
```python
INPUTS = [
    ([1, 2, 3], 2),
    ([], 0),
]
```
""".strip()

DISAGREEMENT_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

You wrote several candidate implementations. They disagree on these inputs, which means the description admits different readings:

{table}

Before you finalize the code you may ask the user exactly ONE clarifying question; there is no second chance.
Pick the disagreement that matters most for ordinary, valid inputs (ignore differences in error handling for invalid inputs) and ask about the underlying requirement in one short, concrete question about exactly one fact. You may cite one of the inputs above with its alternative outputs (e.g. "Should f([3, 1, 3]) return 2 or 3?").

Answer in exactly this format:
ANALYSIS: <1-3 sentences: which readings the outputs correspond to>
QUESTION: <the question>

Never ask about implementation details or test cases, and do not bundle several questions. Write the question in plain text: no Markdown, no backticks.
""".strip()

# Okanagan's regeneration prompt (baseline wording), used for integrate="okanagan".
OKANAGAN_REGEN_PROMPT = """
{prompt}
{clarification}

Given the above conversations, generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`:
"""

FINAL_PROMPT = """
Implement `{entry_point}` for the task below.

<problem>
{prompt}
</problem>

The user was asked one clarifying question and replied:
Question: {question}
Answer: {answer}

The answer states the user's intent: where it conflicts with the task text or its examples, follow the answer; everything the answer does not mention follows the task text. Keep the function name `{entry_point}`.
Enclose your solution in ```python and ```.
""".strip()

REWRITE_PROMPT = """
Task description for `{entry_point}`:

<problem>
{prompt}
</problem>

Clarification from the user:
Question: {question}
Answer: {answer}

Rewrite the task description into one complete and consistent specification that incorporates the user's answer. Where the answer conflicts with the original text or its examples, the answer wins; keep everything else, including the function name `{entry_point}`, its parameters, and the examples that are consistent with the answer (fix or drop the others). Output only the rewritten specification.
""".strip()

FROM_SPEC_PROMPT = """
Implement `{entry_point}` according to this specification:

<specification>
{spec}
</specification>

For reference, the user's original wording was:
<original>
{prompt}
</original>

Follow the specification where the two differ. Enclose your solution in ```python and ```.
""".strip()

TESTS_PROMPT = """
Task for `{entry_point}`:

<problem>
{prompt}
</problem>
{clarification}
Write up to {num_tests} assert statements that check `{entry_point}` on concrete inputs whose expected output follows unambiguously from the task text, its examples, or the clarification above. Do not invent behavior that is not stated; skip a case rather than guess.
Return only a Python code block with one `assert {entry_point}(...) == ...` statement per line.
""".strip()

# Runs several candidates on several inputs inside ONE exec_code call and prints JSON.
HARNESS = r"""
import copy, json, signal, sys, time

class CallTimedOut(BaseException):
    # a BaseException, so candidate code with `except Exception` cannot swallow it
    pass

def on_timer(signum, frame):
    raise CallTimedOut()

def stop_timer():
    signal.alarm(0)

signal.signal(signal.SIGALRM, on_timer)
CANDIDATES = json.loads(CANDIDATES_JSON)
ENTRY = ENTRY_JSON
MODE = MODE_JSON
PAYLOAD = json.loads(PAYLOAD_JSON)

def load_candidate(code):
    ns = {"__name__": "candidate_module"}
    signal.alarm(5)
    try:
        exec(code, ns)
    finally:
        stop_timer()
    return ns

rows = []
inputs = None
# stay well inside the 30 s container timeout, so the results of all candidates are printed
DEADLINE = time.time() + 20
if MODE == "probe":
    ins = {}
    try:
        signal.alarm(5)
        exec(PAYLOAD, ins)
        stop_timer()
        inputs = list(ins["INPUTS"])[:20]
    except BaseException as e:
        stop_timer()
        inputs = []
for code in CANDIDATES:
    row = []
    try:
        ns = load_candidate(code)
        fn = ns[ENTRY]
    except BaseException as e:
        rows.append(None)
        continue
    if MODE == "calls":
        for expr in PAYLOAD:
            try:
                signal.alarm(2)
                out = eval(expr, ns)
                stop_timer()
                row.append(repr(out)[:120])
            except BaseException as e:
                stop_timer()
                row.append("raises " + type(e).__name__)
    elif MODE == "checks":
        # each candidate gets an equal share of the time; after one timeout the rest is skipped
        stop_at = min(DEADLINE, time.time() + 20 / max(1, len(CANDIDATES)))
        hung = False
        for stmt in PAYLOAD["asserts"]:
            if hung or time.time() > stop_at:
                row.append(0)
                continue
            try:
                signal.alarm(2)
                exec(stmt, ns)
                stop_timer()
                row.append(1)
            except CallTimedOut:
                stop_timer()
                hung = True
                row.append(0)
            except BaseException:
                stop_timer()
                row.append(0)
        outs = []
        for expr in PAYLOAD["calls"]:
            if hung or time.time() > stop_at:
                outs.append("skipped")
                continue
            try:
                signal.alarm(2)
                outs.append(repr(eval(expr, ns))[:120])
                stop_timer()
            except CallTimedOut:
                stop_timer()
                hung = True
                outs.append("skipped")
            except BaseException as e:
                stop_timer()
                outs.append("raises " + type(e).__name__)
        row = {"passed": row, "outs": outs}
    elif MODE == "probe":
        for inp in inputs:
            args = inp if isinstance(inp, tuple) else (inp,)
            try:
                signal.alarm(2)
                out = fn(*copy.deepcopy(args))
                stop_timer()
                row.append(repr(out)[:120])
            except BaseException as e:
                stop_timer()
                row.append("raises " + type(e).__name__)
    else:
        for stmt in PAYLOAD:
            try:
                signal.alarm(2)
                exec(stmt, ns)
                stop_timer()
                row.append(1)
            except BaseException:
                stop_timer()
                row.append(0)
    rows.append(row)
result = {"inputs": [repr(i)[:120] for i in (inputs or [])], "rows": rows}
sys.stdout.write("@@HARNESS@@" + json.dumps(result) + "\n")
"""


def extract_code(response: str) -> str | None:
    """Code inside the first ```python ... ``` block (same rule as the SDK's result parser)."""
    if "```python" not in response:
        return None
    _, rest = response.split("```python", 1)
    if "```" not in rest:
        return None
    return rest.split("```", 1)[0]


def escape_closing_tags(text: str) -> str:
    """Escape rich-markup closing tags such as `[/-]` (e.g. inside a regex character class):
    the harness prints every prompt with rich, which raises on an unmatched closing tag."""
    return re.sub(r"(\\*)(\[/[^\[]*?\])", lambda m: m.group(1) * 2 + "\\" + m.group(2), text)


def call_llm(env: ClarificationEnvironment, messages: list[dict[str, str]] | str) -> str:
    """env.llm with up to two retries on errors other than the budget limit; retries escape
    rich-markup closing tags in the last message, the only text the harness prints."""
    attempts = 3
    for attempt in range(attempts):
        try:
            return env.llm(messages)
        except LimitsExceededException:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            if isinstance(messages, str):
                messages = escape_closing_tags(messages)
            else:
                last = dict(messages[-1])
                last["content"] = escape_closing_tags(last["content"])
                messages = messages[:-1] + [last]
    raise AssertionError("unreachable")


def as_code(text: str) -> str:
    """Code from a response that may hold an unclosed ```python fence (e.g. a truncated answer)."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    return text


def after_tag(text: str, tag: str) -> str:
    idx = text.rfind(tag)
    if idx < 0:
        return ""
    return text[idx + len(tag) :].strip()


def norm_name(name: str) -> str:
    return name.lower().replace("_", "")


def sanitize_code(code: str, entry: str, prompt: str) -> str:
    """Drop `if __name__ == "__main__"` blocks and top-level print/input/main calls, and alias
    `entry` to the implemented function when the draft used another name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    drop = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                drop.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in ("print", "input", "main"):
                drop.append((node.lineno, node.end_lineno))
    for start, end in sorted(drop, reverse=True):
        del lines[start - 1 : end]
    code = "\n".join(lines) + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    defined = set()
    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
    if entry in defined or not funcs:
        return code
    names = [f.name for f in funcs]
    chosen = None
    for name in names:
        if norm_name(name) == norm_name(entry):
            chosen = name
    if chosen is None:
        prompt_defs = set(re.findall(r"def\s+(\w+)\s*\(", prompt))
        for name in reversed(names):
            if name in prompt_defs:
                chosen = name
                break
    if chosen is None:
        called = set()
        for f in funcs:
            for sub in ast.walk(f):
                if isinstance(sub, ast.Name):
                    called.add(sub.id)
        roots = [f.name for f in funcs if f.name not in called - {f.name}]
        chosen = roots[-1] if roots else names[-1]
    return code + f"\n\n{entry} = {chosen}\n"


def lhs_call(stmt: str, entry: str) -> str | None:
    """The left-hand call of `assert entry(...) == expected`, as source text."""
    try:
        node = ast.parse(stmt).body[0]
    except (SyntaxError, IndexError):
        return None
    test = getattr(node, "test", None)
    if not (
        isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
    ):
        return None
    left = test.left
    if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == entry:
        return ast.unparse(left)
    return None


def plain_question(text: str | None) -> str | None:
    if not text:
        return None
    text = text.replace("`", "").strip()
    return text or None


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text


class ZarutskiyClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "ask_policy": "always",  # "gate": the question prompt may opt out (Okanagan); "always": always ask
        "question_style": "targeted",  # "okanagan" | "targeted" | "disagreement" | "multi"
        "integrate": "final",  # "okanagan" (baseline regen prompt) | "final" | "rewrite"
        "num_candidates": 3,  # candidates sampled for disagreement probing
        "num_probes": 10,  # probe inputs requested for disagreement probing
        "max_questions": 3,  # cap for question_style="multi"
        "num_final": 1,  # final samples; > 1 enables execution-based selection
        "num_tests": 8,  # self-written asserts for execution-based selection
        "generation_attempts": 3,  # retries when a response has no ```python block
        "sanitize": True,  # entry-point alias + drop __main__ blocks / top-level prints
        "plain_questions": True,  # strip backticks (the simulator's ANSWERS regex stops at one)
        "exact_name": True,  # ask for the exact entry-point name in code prompts
        "interface_hint": False,  # signature-less prompts: point the question at the call interface
        "interface_question": False,  # signature-less prompts: ask which arguments the function takes
        "assume_framing": True,  # P05: phrase the question as an assumption to confirm or correct
        "assume_noY": False,  # X11: the assumption without the "or should it be Y?" alternative
        "exec_assumption": False,  # XE: state the assumption as the draft's real output on one call
        "answer_checks": False,  # SEL: asserts from examples + answer, output agreement, repair
        "include_draft": False,  # SEL: the pre-answer draft joins the final candidates (lowest priority)
        "repair_rounds": 0,  # SEL: repairs of the selected program on failing checks
        "readback": True,  # CR: the question reads the draft's contract back, with an executed example
        "readback_open_close": True,  # CRc: end the read-back with the open close (READBACK_CLOSE_OPEN)
        "interface_assume": False,  # IFA: signature-less prompts: confirm the call interface instead
    }

    def cfg(self, key: str) -> Any:
        return self.config.get(key, self.DEFAULT_CONFIG[key])

    # -- generation helpers -------------------------------------------------

    def generate_code(self, env: ClarificationEnvironment, content: str) -> tuple[str, str]:
        """Returns (code or "", last raw response)."""
        messages = [{"role": "user", "content": content}]
        response = ""
        for _ in range(max(1, int(self.cfg("generation_attempts")))):
            response = call_llm(env, messages)
            code = extract_code(response)
            if code is not None and code.strip():
                return code, response
            messages = messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "Please answer with the complete implementation enclosed in ```python and ```.",
                },
            ]
        return "", response

    def code_prompt(self, problem) -> str:
        rule = "Name the function exactly `{entry_point}`. " if self.cfg("exact_name") else ""
        return (
            CODE_PROMPT.replace("{name_rule}", rule)
            .replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
        )

    def seed_candidate(self, env, problem) -> str:
        content = self.code_prompt(problem)
        code, raw = self.generate_code(env, content)
        return code or as_code(raw)

    def run_harness(self, env, candidates: list[str], entry: str, mode: str, payload: Any) -> dict:
        script = (
            "CANDIDATES_JSON = "
            + repr(json.dumps(candidates))
            + "\nENTRY_JSON = "
            + repr(entry)
            + "\nMODE_JSON = "
            + repr(mode)
            + "\nPAYLOAD_JSON = "
            + repr(json.dumps(payload))
            + "\n"
            + HARNESS
        )
        try:
            out = env.exec_code(script)
        except Exception:
            return {}
        idx = out.rfind("@@HARNESS@@")
        if idx < 0:
            return {}
        try:
            return json.loads(out[idx + len("@@HARNESS@@") :].strip().splitlines()[0])
        except (ValueError, IndexError):
            return {}

    # -- question styles ------------------------------------------------------

    def question_okanagan(self, env, problem, candidate) -> str | None:
        template = (
            OKANAGAN_QUESTION_PROMPT if self.cfg("ask_policy") == "gate" else OKANAGAN_ALWAYS_PROMPT
        )
        response = call_llm(
            env, template.replace("{prompt}", problem["prompt"]).replace("{candidate}", candidate)
        )
        if "```" in response:
            _, response = response.split("```", 1)
        if "```" in response:
            response, _ = response.split("```", 1)
        if self.cfg("ask_policy") == "gate" and "NO_QUESTIONS" in response:
            return None
        return response.strip() or None

    def question_targeted(self, env, problem, candidate, allow_exec=True) -> str | None:
        hint = ""
        if self.cfg("interface_hint") and not re.search(r"def\s+\w+\s*\(", problem["prompt"]):
            hint = INTERFACE_HINT.replace("{entry_point}", problem["entry_point"])
        template = TARGETED_QUESTION_PROMPT
        z0_line = 'QUESTION: <one short question about exactly one fact, answerable with a single value or choice. When it helps, name the concrete alternatives or a tiny example input, e.g. "Should f([3, 1, 3]) return 2 or 3?">'
        no_def = not re.search(r"def\s+\w+\s*\(", problem["prompt"])
        use_exec = self.cfg("exec_assumption") and allow_exec
        if self.cfg("interface_assume") and no_def:
            template = template.replace(z0_line, IFACE_ASSUME_TEXT)
            use_exec = False
        elif use_exec:
            template = template.replace(z0_line, EXEC_ASSUME_TEXT)
        elif self.cfg("assume_noY") or self.cfg("exec_assumption"):
            template = template.replace(z0_line, ASSUME_NOY_TEXT)
        elif self.cfg("assume_framing"):
            template = template.replace(z0_line, ASSUME_FRAMING_TEXT)
        response = call_llm(
            env,
            template.replace("{interface_hint}", hint)
            .replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{candidate}", candidate),
        )
        question = strip_fences(after_tag(response, "QUESTION:"))
        if not question:
            question = response.strip().splitlines()[-1] if response.strip() else ""
        if use_exec and "RESULT" in question:
            question = self.fill_result(env, problem, candidate, response, question)
        return question or None

    def field(self, response: str, tag: str) -> str:
        for line in response.splitlines():
            line = line.strip().replace("`", "")
            if line.upper().startswith(tag + ":"):
                return line[len(tag) + 1 :].strip().rstrip(".")
        return ""

    def question_readback(self, env, problem, candidate) -> str | None:
        entry = problem["entry_point"]
        response = call_llm(
            env,
            READBACK_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{candidate}", candidate),
        )
        args, output, assumption = (
            self.field(response, t) for t in ("ARGS", "OUTPUT", "ASSUMPTION")
        )
        call = self.field(response, "CALL")
        if not (args and output and assumption):
            return self.question_targeted(env, problem, candidate, allow_exec=False)
        example = ""
        if call.startswith(entry + "(") and len(call) <= 120:
            result = self.run_harness(env, [candidate], entry, "calls", [call])
            rows = result.get("rows") or [None]
            out = rows[0][0] if rows[0] else None
            if out and not out.startswith("raises ") and len(out) <= 80:
                example = f" So {call} would return {out}."
        close = READBACK_CLOSE_OPEN if self.cfg("readback_open_close") else READBACK_CLOSE
        return (
            READBACK_TEMPLATE.replace("{entry_point}", entry)
            .replace("{args}", args)
            .replace("{output}", output)
            .replace("{assumption}", assumption[:1].lower() + assumption[1:])
            .replace("{example}", example)
            .replace("{close}", close)
        )

    def fill_result(self, env, problem, candidate, response, question) -> str | None:
        """Replace RESULT with what the draft returns on the CALL line; else ask without the call."""
        entry = problem["entry_point"]
        call = response[response.rfind("CALL:") + len("CALL:") :] if "CALL:" in response else ""
        call = strip_fences(
            call.split("QUESTION:", 1)[0].strip().splitlines()[0] if call.strip() else ""
        )
        call = call.replace("`", "").strip()
        out = None
        if call.startswith(entry + "(") and len(call) <= 200:
            result = self.run_harness(env, [candidate], entry, "calls", [call])
            rows = result.get("rows") or [None]
            if rows[0]:
                out = rows[0][0]
        if out is None or len(out) > 80 or out.startswith("raises "):
            # no usable execution: fall back to the plain assumption question
            return self.question_targeted(env, problem, candidate, allow_exec=False)
        return question.replace("RESULT", out)

    def question_multi(self, env, problem, candidate) -> str | None:
        response = call_llm(
            env,
            MULTI_QUESTION_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{candidate}", candidate)
            .replace("{max_questions}", str(self.cfg("max_questions"))),
        )
        block = after_tag(response, "QUESTIONS:")
        lines = [ln.strip() for ln in block.splitlines() if re.match(r"^\s*\d+[.)]\s+\S", ln)]
        lines = lines[: max(1, int(self.cfg("max_questions")))]
        return "\n".join(lines) if lines else None

    def question_disagreement(self, env, problem, candidate) -> tuple[str | None, dict]:
        entry = problem["entry_point"]
        base = self.code_prompt(problem)
        candidates = [candidate]
        for _ in range(max(0, int(self.cfg("num_candidates")) - 1)):
            code, _ = self.generate_code(env, base)
            if code:
                candidates.append(code)
        probe_resp = call_llm(
            env,
            PROBE_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{num_probes}", str(self.cfg("num_probes"))),
        )
        probe_code = extract_code(probe_resp) or ""
        info = {"num_candidates": len(candidates), "disagreements": 0}
        if not probe_code or len(candidates) < 2:
            return None, info
        result = self.run_harness(env, candidates, entry, "probe", probe_code)
        rows = [r for r in result.get("rows", []) if r is not None]
        inputs = result.get("inputs", [])
        if len(rows) < 2 or not inputs:
            return None, info
        valid_lines, error_lines = [], []
        for j, inp in enumerate(inputs):
            outs = [r[j] for r in rows if j < len(r)]
            distinct = sorted(set(outs))
            if len(distinct) > 1:
                info["disagreements"] += 1
                call = entry + (inp if inp.startswith("(") else "(" + inp + ")")
                line = f"- {call}: " + " vs. ".join(distinct)
                if any(o.startswith("raises ") for o in outs):
                    error_lines.append(line)
                else:
                    valid_lines.append(line)
        # disagreements on valid inputs first: tests exercise valid inputs, not error handling
        table_lines = (valid_lines + error_lines)[:5]
        if not table_lines:
            return None, info
        response = call_llm(
            env,
            DISAGREEMENT_QUESTION_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{table}", "\n".join(table_lines)),
        )
        question = strip_fences(after_tag(response, "QUESTION:"))
        return (question or None), info

    # -- integration ------------------------------------------------------------

    def final_code(self, env, problem, question: str | None, answer: str | None) -> list[str]:
        entry = problem["entry_point"]
        n = max(1, int(self.cfg("num_final")))
        if question is None:
            content = self.code_prompt(problem)
        elif self.cfg("integrate") == "okanagan":
            clar = f"Questions #1:\n{question}\nAnswers:\n{answer}\n"
            content = (
                OKANAGAN_REGEN_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{clarification}", clar)
                .replace("{entry_point}", entry)
            )
        elif self.cfg("integrate") == "rewrite":
            spec = call_llm(
                env,
                REWRITE_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
                .replace("{question}", question)
                .replace("{answer}", answer or ""),
            ).strip()
            content = (
                FROM_SPEC_PROMPT.replace("{spec}", spec)
                .replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
            )
        else:
            content = (
                FINAL_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
                .replace("{question}", question)
                .replace("{answer}", answer or "")
            )
        codes = []
        last_raw = ""
        for _ in range(n):
            code, raw = self.generate_code(env, content)
            last_raw = raw
            if code:
                codes.append(code)
        return codes or [as_code(last_raw)]

    def select(self, env, problem, codes: list[str], question, answer) -> str:
        if len(codes) == 1:
            return codes[0]
        entry = problem["entry_point"]
        clar = ""
        if question:
            clar = f"\nClarification from the user:\nQuestion: {question}\nAnswer: {answer}\n"
        resp = call_llm(
            env,
            TESTS_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{clarification}", clar)
            .replace("{num_tests}", str(self.cfg("num_tests"))),
        )
        block = extract_code(resp) or ""
        asserts = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("assert ")]
        if not asserts:
            return codes[0]
        result = self.run_harness(env, codes, entry, "asserts", asserts)
        rows = result.get("rows", [])
        if len(rows) != len(codes):
            return codes[0]
        scores = [sum(r) if r is not None else -1 for r in rows]
        best = max(scores)
        tied = [i for i, sc in enumerate(scores) if sc == best]
        # tie-break by agreement: prefer the pass/fail pattern shared by most candidates
        patterns = [tuple(rows[i]) if rows[i] is not None else () for i in tied]
        counts = {pat: patterns.count(pat) for pat in patterns}
        top = max(tied, key=lambda i: counts[tuple(rows[i]) if rows[i] is not None else ()])
        return codes[top]

    # -- main ---------------------------------------------------------------------

    def clean(self, code: str, problem: dict[str, Any]) -> str:
        if not self.cfg("sanitize"):
            return code
        return sanitize_code(code, problem["entry_point"], problem["prompt"])

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        candidate = self.clean(self.seed_candidate(env, problem), problem)
        try:
            return self.refine(env, problem, candidate)
        except Exception:
            # any failure after the seed (budget, harness console error) keeps the seed draft
            return candidate

    def refine(self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str) -> str:
        question, answer = None, None
        if env.can_ask():
            style = self.cfg("question_style")
            if self.cfg("interface_question") and not re.search(
                r"def\s+\w+\s*\(", problem["prompt"]
            ):
                # the call interface is the first thing a signature-less draft gets wrong
                style = "interface"
            if style == "interface":
                question = INTERFACE_QUESTION.replace("{entry_point}", problem["entry_point"])
            elif style == "okanagan":
                question = self.question_okanagan(env, problem, candidate)
            elif style == "multi":
                question = self.question_multi(env, problem, candidate)
            elif style == "disagreement":
                question, probe_info = self.question_disagreement(env, problem, candidate)
                if question is None and self.cfg("ask_policy") == "always":
                    question = self.question_targeted(env, problem, candidate)
            elif self.cfg("readback"):
                question = self.question_readback(env, problem, candidate)
            else:
                question = self.question_targeted(env, problem, candidate)
            if self.cfg("plain_questions"):
                question = plain_question(question)
            if question is not None:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    question, answer = None, None
        if question is None and int(self.cfg("num_final")) <= 1:
            return candidate
        codes = [self.clean(c, problem) for c in self.final_code(env, problem, question, answer)]
        if self.cfg("answer_checks"):
            if self.cfg("include_draft") and candidate.strip():
                codes = codes + [candidate]
            try:
                return self.select_checks(env, problem, codes, question, answer)
            except Exception:
                # budget or harness trouble during selection keeps the first final
                return codes[0]
        return self.select(env, problem, codes, question, answer)

    def write_checks(self, env, problem, question, answer) -> list[str]:
        clar = ""
        if question:
            clar = f"\nClarification from the user:\nQuestion: {question}\nAnswer: {answer}\n"
        resp = call_llm(
            env,
            TESTS_SEL_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{clarification}", clar)
            .replace("{num_tests}", str(self.cfg("num_tests"))),
        )
        block = extract_code(resp) or ""
        asserts = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("assert ") or line in asserts:
                continue
            try:
                ast.parse(line)
            except SyntaxError:
                continue
            asserts.append(line)
        return asserts[:12]

    def run_checks(self, env, codes, entry, asserts, calls) -> list[dict | None]:
        result = self.run_harness(env, codes, entry, "checks", {"asserts": asserts, "calls": calls})
        rows = result.get("rows", [])
        if len(rows) != len(codes):
            return [None] * len(codes)
        return rows

    def select_checks(self, env, problem, codes: list[str], question, answer) -> str:
        """Pick the candidate passing most answer-bound checks; ties go to the candidate whose
        outputs on the checked calls agree with most others; then one guarded repair."""
        entry = problem["entry_point"]
        asserts = self.write_checks(env, problem, question, answer)
        if not asserts:
            return codes[0]
        calls = [lhs_call(a, entry) for a in asserts]
        rows = self.run_checks(env, codes, entry, asserts, [c or "None" for c in calls])
        # a check no candidate passes is more likely wrong than all candidates (filter as in CrossCheck)
        all_asserts, all_calls = asserts, calls
        full = [dict(r) if r else None for r in rows]
        keep = [k for k in range(len(asserts)) if any(r and r["passed"][k] for r in rows)]
        asserts, calls = [asserts[k] for k in keep], [calls[k] for k in keep]
        for r in rows:
            if r:
                r["passed"] = [r["passed"][k] for k in keep]
                r["outs"] = [r["outs"][k] for k in keep]
        scores = [sum(r["passed"]) if r else -1 for r in rows]
        if max(scores) < 0:
            return codes[0]
        best = max(scores)
        tied = [i for i, sc in enumerate(scores) if sc == best]

        def agreement(i):
            outs = rows[i]["outs"]
            return sum(
                1
                for j, other in enumerate(rows)
                if j != i and other
                for k, out in enumerate(other["outs"])
                if calls[k] and k < len(outs) and out == outs[k] and out != "skipped"
            )

        top = max(tied, key=lambda i: (agreement(i), -i))
        chosen = codes[top]
        # the guarded repair sees every check, including those no candidate passes
        asserts, calls = all_asserts, all_calls
        chosen_score = sum(full[top]["passed"])
        if chosen_score < len(asserts) and int(self.cfg("repair_rounds")) > 0:
            outs = full[top]["outs"]
            failures = []
            for k, (stmt, ok) in enumerate(zip(asserts, full[top]["passed"], strict=False)):
                if not ok:
                    got = f" (your code gives {outs[k]})" if calls[k] and k < len(outs) else ""
                    failures.append(f"- {stmt}{got}")
            clar = ""
            if question:
                clar = f"\nClarification from the user:\nQuestion: {question}\nAnswer: {answer}\n"
            # one repair at most; kept only if it passes strictly more checks
            code, _ = self.generate_code(
                env,
                REPAIR_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
                .replace("{clarification}", clar)
                .replace("{code}", chosen)
                .replace("{failures}", "\n".join(failures[:6])),
            )
            if code:
                code = self.clean(code, problem)
                new = self.run_checks(env, [code], entry, asserts, [])[0]
                if new and sum(new["passed"]) > chosen_score:
                    chosen = code
        return chosen
