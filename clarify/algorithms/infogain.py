# SPDX-FileCopyrightText: 2026 Md Zahidul Haque <mhaque@wm.edu>
#
# SPDX-License-Identifier: MIT

"""InfoGain clarifier.

Ask about the point where two careful readers of the task would write different
code and only one of them would pass: how the function is called and what it
returns, a vague phrase, a contradiction, bounds, positions, order and so on.
A few programs are sampled and run on the same inputs; where their outputs split,
that is passed along as a hint about where the task is open. The final program
is picked with asserts written from the answer.

Tasks that show no def line are handled a little differently: statements that
cannot hold together with the rest are dropped first, and the question is
steered towards how the function is called, since that is what such tasks most
often leave open. Tasks with a def line go through the plain path.

A reply that settles nothing ("not specified", "cannot answer") is not used to
rewrite the program; the programs from before the question are kept instead.

Team: RankOne
Team Members: Md Zahidul Haque
Main Contact: mhaque@wm.edu
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from collections import Counter
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

CODE_TEMPLATE = """
Write a self-contained Python solution for the task below.

{prompt}
{context}
The tests call a top-level function (or class, if the task describes one) named `{entry_point}`. Use
exactly that name, even if the task text calls it something else. Only define it and any helpers: no
`__main__` block, prints, input() or asserts. Do not access files, the network or the shell, and do not use eval or exec.
Put the code in ```python and ```.
""".strip()

PROBE_TEMPLATE = """
Task (implement `{entry_point}`):
{prompt}

Write {num_probes} diverse argument lists for calling `{entry_point}` that would expose differences
between plausible interpretations of the task (edge cases, boundary values, unusual formats).
Return ONE Python literal: a list where each element is a list of positional arguments.
Enclose it in ```python and ```. Use only literals (no function calls, no imports).
""".strip()

QUESTION_TEMPLATE = """
Task (the tests call `{entry_point}`):
{prompt}

You may ask the user one question before implementing this. Picture two careful developers who each
implement it from the text alone. Find the point where they are most likely to write different code,
so that only one of them would pass the user's tests. It is usually one of these:
- how the function is called and what it returns: the arguments, their order and types, and the
  exact type and format of the result (list or tuple, int or float, exact strings)
- what a vague or unusual phrase in the task means exactly
- a statement that disagrees with another statement or with an example
- limits: inclusive or exclusive bounds, strict or non-strict comparisons
- positions: 0-based or 1-based, counted from the start or the end
- empty and edge inputs, the order of the result, duplicates and ties
- text: case, spaces and punctuation; numbers: rounding, precision, integer or float
- tabular or array data: missing values (NaN/None), rows or columns (axis), keep or reset the index,
  column order, and whether to return a DataFrame, Series, array or plain list
Skip anything the task or its examples already settle, and skip input validation, error handling and
performance: the tests use ordinary valid inputs.
{call}{evidence}
Ask about that one point in a single sentence with one question mark, in plain words and without
code. Either name the two most likely choices ("Should ... be A or B?") or ask what the phrase means
("What exactly does ... mean here?"). Never add an open option such as "or something else". {style} Some point is always worth confirming,
so never reply that nothing is open.

{quote}End with one line of the form
QUESTION: <your question>
""".strip()

PLAIN_STYLE = "Do not\nask for test cases or for the output of a particular input."

WITNESS_STYLE = """When a small concrete call shows the difference, ask it that way instead: "For
`{entry_point}(...)`, should the result be A or B?", with an input on which the two readings give
different results."""

QUOTE_NOTE = """Just before it, add one line QUOTE: <the words of the task your question is about, copied
exactly>{none}.
"""

QUOTE_NONE = ", or QUOTE: none if the question is about how the function is called"

QUOTE_RETRY = """
Your last question pointed at "{quote}", which is not in the task. Ask about something the task
actually says, and copy those words exactly in the QUOTE line.
"""

CALL_NOTE = """
This task gives no function signature, so the way the tests call `{entry_point}` is the point most
likely to differ: how many arguments, in what order and of what type, and what exactly is returned.
Unless the text already settles all of that, ask about the call, naming the most likely form, for
example "Should `{entry_point}` take (a, b) in that order and return a list?". The call counts as one
point.
"""

HINT = """
Notes (the user has not seen these): a few drafts were run on the same inputs and disagreed here.
Take it as a hint about where the task is open, not as the question itself; the fact that matters
most may be elsewhere.
{lines}
"""

ANSWER_CONTEXT = """
The user was asked: {question}
They answered: {answer}
The answer is the final word on that point. If it disagrees with the task text, go with the answer;
everything else follows the task.
"""

ANSWER_TESTS_TEMPLATE = """
Task (implement `{entry_point}`):
{prompt}

The user clarified:
Q: {question}
A: {answer}

Write 3-5 Python assert statements that call `{entry_point}` and check ONLY the behaviour fixed by this
clarification. Do not define `{entry_point}`. Enclose them in ```python and ```.
""".strip()

CLEAN_TEMPLATE = """
Below is a programming task. Some tasks in this collection were edited by adding statements that
cannot hold together with the rest, for example: an impossible limit such as O(1) time or "no loops"
for a task that has to look at every element, a demand to return numbers or booleans as strings, a
demand to also accept a different kind of input, a closing "but return X instead" that undoes the main
sentence, or an example whose output disagrees with the description.

If the task has such statements, rewrite it without them. Keep what the function name, the signature,
the main description and most of the examples agree on; if a single example disagrees with that, drop
the example. Keep everything else word for word, including the signature and the other examples, and
do not add anything.

A statement that agrees with the signature, its return annotation or the examples is not a clash, and
neither is anything that is only vague or missing. Choices the task leaves open, such as "in ascending or
descending order" or "the first or the last match", are not clashes either: keep them exactly as written and
do not pick one. If nothing in the task clashes, reply with the
single word UNCHANGED. Most tasks are like that.

Task:
{prompt}

Reply with UNCHANGED, or with the rewritten task inside ```text and ```.
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
    # re-arm every call, a candidate may have reset the handler or the alarm
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
    kind = type(value).__module__.split(".")[0]
    if kind == "numpy" and hasattr(value, "tolist"):
        # scalars compare equal to plain numbers under ==, arrays do not compare like lists
        if getattr(value, "ndim", 0) == 0:
            return _canon(value.tolist())
        return type(value).__name__ + ":" + _canon(value.tolist())
    if kind == "pandas" and hasattr(value, "to_dict"):
        # full contents in order; repr would cut long frames
        rows = value.to_dict("split") if hasattr(value, "columns") else list(value.items())
        return type(value).__name__ + ":" + _canon(rows)
    if hasattr(value, "__next__"):
        return "iter" + _canon(list(itertools.islice(value, 1000)))
    text = repr(value)
    if " at 0x" not in text:
        return text
    if hasattr(value, "__dict__") and not callable(value):
        return type(value).__name__ + _canon(vars(value))
    return type(value).__name__


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
    # re-arm every call, a candidate may have reset the handler or the alarm
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

# the sandbox kills the process at 30s, so stop early
EXEC_DEADLINE_SECONDS = 15


def extract_block(text: str, lang: str) -> str | None:
    match = re.search(rf"```{lang}\s*\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1) if match else None


def defined_names(tree: ast.Module) -> set[str]:
    defs = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    names = {n.name for n in tree.body if isinstance(n, defs)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def sets_recursion_limit(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "setrecursionlimit"
    )


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
    tree.body = [node for node in tree.body if isinstance(node, keep) or sets_recursion_limit(node)]
    if entry_point not in defined_names(tree):
        return code
    return ast.unparse(tree)


UNSAFE_MODULES = {
    "os",
    "subprocess",
    "socket",
    "shutil",
    "requests",
    "urllib",
    "http",
    "ctypes",
    "pickle",
    "marshal",
    "multiprocessing",
    "threading",
    "signal",
    "importlib",
    "pathlib",
    "tempfile",
    "glob",
    "asyncio",
    "ftplib",
    "smtplib",
    "webbrowser",
    "resource",
    "pty",
    "builtins",
}
UNSAFE_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}


def unsafe_reason(code: str) -> str | None:
    """Why generated code touches files, network, shell or dynamic execution; None if it doesn't."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in UNSAFE_MODULES:
                return f"imports {name}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in UNSAFE_CALLS:
                return f"calls {node.func.id}()"
    return None


RANDOM_MODULES = {"random", "secrets", "uuid", "time"}


def nondeterministic(code: str) -> bool:
    """Programs whose output can change from run to run make useless evidence."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] in RANDOM_MODULES for name in names):
            return True
    return False


def bind_entry_point(code: str, entry_point: str, prompt: str) -> str:
    """If the model named the function differently, alias it to the name the tests call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if entry_point in defined_names(tree):
        return code
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            isinstance(m, ast.FunctionDef) and m.name == entry_point for m in node.body
        ):
            return f"{code.rstrip()}\n\n{entry_point} = {node.name}().{entry_point}\n"
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if not funcs:
        return code
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    roots = [f for f in funcs if f not in called] or funcs
    in_prompt = [f for f in roots if re.search(rf"\bdef\s+{re.escape(f)}\s*\(", prompt)]
    target = (in_prompt or roots)[-1]
    return f"{code.rstrip()}\n\n{entry_point} = {target}\n"


SLASH_HINT = (
    "\nIn regular expressions, do not put a slash right after an opening square bracket; put it last "
    "in the character class."
)

# replies that settle nothing: vague refusals, and the two fixed replies of the simulated user
NON_ANSWER = (
    r"\b(?:not specified|unspecified|can'?t confirm|cannot confirm|can'?t answer|cannot answer|"
    r"not mentioned|no information|irrelevant question)\b"
)


def says_nothing(answer: str | None) -> bool:
    return bool(re.search(NON_ANSWER, answer or "", flags=re.IGNORECASE))


def unprintable(error: Exception) -> bool:
    # the harness prints every message with rich markup on; "[/x]" in it reads as a closing tag
    return "MarkupError" in repr(type(error))


CATCH_ALL = r"(?:something|anything)\s+else(?:\s+entirely)?"


COMPOUND = (
    r"[,;]?\s+and\s+(?:also\s+)?"
    r"(?:should|does|do|is|are|can|will|would|must|what|how|which|where|when)\b[^?]*\?"
)


def tidy_question(text: str, cut_compound: bool = False) -> str | None:
    """One line, one question, no "or something else" tail; None if it is not a question."""
    question = " ".join((text or "").split())
    # "A, B, or something else?" -> "A or B?"
    trimmed = re.sub(
        rf",\s+([^,?]+?),?\s+or\s+{CATCH_ALL}(?=\s*\?)", r" or \1", question, flags=re.IGNORECASE
    )
    trimmed = re.sub(rf",?\s+or\s+{CATCH_ALL}(?=\s*\?)", "", trimmed, flags=re.IGNORECASE)
    trimmed = re.sub(r"\s+\?", "?", trimmed)
    if cut_compound:
        # "..., and should Y?" asks two things; keep the first
        trimmed = re.sub(COMPOUND, "?", trimmed, count=1, flags=re.IGNORECASE)
    # a second question after the first: keep the first, but not a "?" inside quotes
    for match in re.finditer(r"\?(?=\s)", trimmed):
        before = trimmed[: match.start()]
        if before.count('"') % 2 == 0 and before.count("\u201c") == before.count("\u201d"):
            trimmed = trimmed[: match.end()]
            break
    for candidate in (trimmed, question):
        if "?" in candidate and re.search(r"[^\W\d_]", candidate):
            return candidate
    return None


def parse_question(text: str, cut_compound: bool = False) -> str | None:
    lines = (text or "").splitlines()
    for line in reversed(lines):
        match = re.match(r"^[\s*>`-]*QUESTION\s*:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
        if match:
            return tidy_question(match.group(1).strip().strip("*`"), cut_compound)
    # no marker: fall back to the last line that is a question
    for line in reversed(lines):
        question = tidy_question(line.strip().strip("*`>-").strip(), cut_compound)
        if question:
            return question
    return None


def parse_quote(text: str) -> str | None:
    for line in reversed((text or "").splitlines()):
        match = re.match(r"^[\s*>`-]*QUOTE\s*:\s*(.*)$", line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("*`")
    return None


def squash(text: str) -> str:
    return " ".join(re.sub(r"[\"'`\u201c\u201d\u2018\u2019]", " ", text or "").lower().split())


def quote_in_task(quote: str, prompt: str) -> bool:
    words = squash(quote)
    return bool(words) and words in squash(prompt)


def has_signature(prompt: str) -> bool:
    return re.search(r"\bdef\s+\w+\s*\(", prompt or "") is not None


def extract_text(text: str) -> str | None:
    """A ```text block up to its last fence, so fences inside the task do not cut it short."""
    match = re.search(r"```text\s*\n(.*)```", text or "", flags=re.DOTALL)
    return match.group(1).strip() if match else None


CHOICE = r"\b(\w+)\s+or\s+(\w+)\b"


def keeps_choices(original: str, rewritten: str) -> bool:
    """Every "X or Y" in the task survives the rewrite: an open choice is not a clash to settle."""
    pairs = {(a.lower(), b.lower()) for a, b in re.findall(CHOICE, original, flags=re.IGNORECASE)}
    kept = {(a.lower(), b.lower()) for a, b in re.findall(CHOICE, rewritten, flags=re.IGNORECASE)}
    return pairs <= kept


def accept_rewrite(original: str, rewritten: str | None, entry_point: str) -> str | None:
    """The cleaned task, or None when the reply is not a plausible trimmed copy of the original."""
    text = (rewritten or "").strip()
    if not text or text.upper().startswith("UNCHANGED"):
        return None
    ratio = len(text) / max(len(original.strip()), 1)
    if not 0.4 <= ratio <= 1.1:
        return None
    signature = rf"\bdef\s+{re.escape(entry_point)}\s*\("
    if re.search(signature, original) and not re.search(signature, text):
        return None
    # a trimmed copy: almost no new words, at most one example dropped, docstring quotes intact
    words = Counter(re.findall(r"\w+", original.lower()))
    added = Counter(re.findall(r"\w+", text.lower())) - words
    if sum(added.values()) > max(3, sum(words.values()) // 10):
        return None
    if text.count(">>>") < original.count(">>>") - 1:
        return None
    if any(text.count(q) % 2 != original.count(q) % 2 for q in ('"""', "'''")):
        return None
    if not keeps_choices(original, text):
        return None
    return text


def entropy(counts: list[float]) -> float:
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c > 0) if total else 0.0


def split_probes(outputs: dict[str, list[str]], limit: int) -> list[int]:
    """Probe indices ordered by how much they split the programs (output entropy), best first."""
    rows = list(outputs.values())
    if len(rows) < 2:
        return []
    width = min(len(r) for r in rows)
    scored = []
    for j in range(width):
        # a crash is a bug in one program, not a second reading of the task
        values = [r[j] for r in rows if not str(r[j]).startswith("ERR:")]
        if len(values) >= 2:
            scored.append((entropy(list(Counter(values).values())), j))
    return [j for gain, j in sorted(scored, key=lambda t: (-t[0], t[1])) if gain > 0][:limit]


def clip(text: str, size: int = 160) -> str:
    return text if len(text) <= size else text[: size - 3] + "..."


class InfoGainClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "num_candidates": 3,  # programs sampled before asking
        "num_probes": 8,  # inputs used to compare them
        "max_evidence": 3,  # disagreeing inputs shown to the question writer
        "num_final": 2,  # programs sampled after the answer
        "use_answer_tests": True,  # pick the final program with asserts from the answer
        "clean_contradictions": True,  # no def line: drop statements that clash with the rest first
        "call_note": True,  # no def line: steer the question towards how the function is called
        "witness": False,  # ask with a concrete call and two possible results
        "evidence_quote": False,  # the question must point at words that are really in the task
        "keep_on_non_answer": True,  # a reply that settles nothing does not replace the programs we have
        "time_limit": 200,  # seconds; past this, skip the optional steps (harness stops at 300)
    }

    # per-task state, reset in run(); the repo validator rejects underscore attributes
    kept = ()  # programs that came back clean during the current task
    deadline = math.inf

    def over_time(self):
        return time.monotonic() > self.deadline

    def chat(self, env, messages):
        """env.llm, except that a reply the harness cannot print comes back empty."""
        try:
            return env.llm(messages)
        except Exception as e:
            if unprintable(e):
                return ""
            raise

    # llm helpers

    def generate(self, env, problem, context=""):
        prompt = (
            CODE_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{context}", context)
        )
        messages = [{"role": "user", "content": prompt}]
        for _ in range(2):
            response = self.chat(env, messages)
            if not response:
                messages = [
                    *messages[:-1],
                    {**messages[-1], "content": messages[-1]["content"] + SLASH_HINT},
                ]
                continue
            try:
                code = _validate_and_parse_evalplus_result(response)
                compile(code, "<solution>", "exec")
            except SyntaxError as e:
                feedback = f"The code does not compile: {e}. Return the corrected code in ```python and ```."
            except ValueError as e:
                feedback = str(e)
            else:
                code = bind_entry_point(code, problem["entry_point"], problem["prompt"])
                code = sanitize(code, problem["entry_point"])
                reason = unsafe_reason(code)
                if reason is None:
                    self.kept = [*self.kept, code]
                    return code
                feedback = (
                    f"The code {reason}. Rewrite it as a plain function without files, network, "
                    "shell, eval or exec. Put the code in ```python and ```."
                )
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": feedback},
            ]
        return None

    def probes(self, env, problem):
        response = self.chat(
            env,
            PROBE_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{num_probes}", str(self.config["num_probes"])),
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

    # execution

    def execute(self, env, template, **fields):
        try:
            output = env.exec_code(template.format(**fields))
        except Exception:  # docker hiccup, skip this step instead of failing the task
            return None
        if "__RESULT__" not in output:
            return None
        try:
            return json.loads(output.rsplit("__RESULT__", 1)[1].strip().splitlines()[0])
        except (json.JSONDecodeError, IndexError):
            return None

    def run_programs(self, env, problem, candidates, probes):
        """Outputs per program on the probes; programs that fail to load are left out."""
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
            return {}
        return {name: outs for name, outs in results.items() if isinstance(outs, list) and outs}

    # question

    def evidence(self, problem, probes, outputs):
        lines = []
        for j in split_probes(outputs, self.config["max_evidence"]):
            call = clip(f"{problem['entry_point']}({', '.join(repr(a) for a in probes[j])})")
            groups = Counter(outs[j] for outs in outputs.values())
            shown = "; ".join(f"{n} program(s) return {clip(v, 80)}" for v, n in groups.items())
            lines.append(f"- {call}: {shown}")
        return "\n".join(lines)

    def write_question(self, env, problem, evidence):
        open_call = not has_signature(problem["prompt"])
        call = CALL_NOTE if open_call and self.config["call_note"] else ""
        hint = HINT.replace("{lines}", evidence) if evidence else ""
        style = WITNESS_STYLE if self.config["witness"] else PLAIN_STYLE
        quote = ""
        if self.config["evidence_quote"]:
            quote = QUOTE_NOTE.replace("{none}", QUOTE_NONE if open_call else "")
        prompt = (
            QUESTION_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{call}", call)
            .replace("{evidence}", hint)
            .replace("{style}", style)
            .replace("{quote}", quote)
            .replace("{entry_point}", problem["entry_point"])
        )
        response = self.chat(env, prompt)
        question = parse_question(response, cut_compound=bool(call))
        if not quote or self.over_time():
            return question
        said = parse_quote(response) or ""
        if quote_in_task(said, problem["prompt"]) or (open_call and said.lower() == "none"):
            return question
        try:
            response = self.chat(env, prompt + QUOTE_RETRY.replace("{quote}", said))
        except LimitsExceededException:
            return question
        return parse_question(response, cut_compound=bool(call)) or question

    def clean_task(self, env, problem):
        response = self.chat(env, CLEAN_TEMPLATE.replace("{prompt}", problem["prompt"]))
        cleaned = accept_rewrite(problem["prompt"], extract_text(response), problem["entry_point"])
        return {**problem, "prompt": cleaned} if cleaned else problem

    # final selection

    def answer_tests(self, env, problem, question, answer):
        response = self.chat(
            env,
            ANSWER_TESTS_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{question}", question)
            .replace("{answer}", answer),
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

    def healthy(self, env, problem, programs, probes):
        """Programs that load and crash least on the probes; None if the sandbox run failed."""
        sources = {f"c{i}": c for i, c in enumerate(programs)}
        results = self.execute(
            env,
            RUNNER,
            sources=repr(sources),
            probes=repr(probes),
            entry=problem["entry_point"],
            deadline=EXEC_DEADLINE_SECONDS,
        )
        if results is None:
            return None
        crashes = {
            name: sum(str(v).startswith("ERR:") for v in outs)
            for name, outs in results.items()
            if isinstance(outs, list) and outs
        }
        if not crashes:
            return []
        fewest = min(crashes.values())
        return [c for i, c in enumerate(programs) if crashes.get(f"c{i}") == fewest]

    def select_final(self, env, problem, finals, tests, probes=None, fallback=None):
        if probes:
            ok = self.healthy(env, problem, finals, probes)
            if ok == [] and fallback:
                return fallback[0]
            if ok:
                finals = ok
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
        # tie-break: most common program, else the first
        return Counter(pool).most_common(1)[0][0]

    # main

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        self.kept = []
        self.deadline = time.monotonic() + self.config["time_limit"]
        try:
            return self.solve(env, problem)
        except LimitsExceededException:
            # out of model budget: the last program that came back clean beats nothing
            return self.kept[-1] if self.kept else ""
        except Exception:
            # anything else (an api error, a reply the harness cannot print): same fallback
            return self.kept[-1] if self.kept else ""

    def sample(self, env, problem, count, context=""):
        """Up to `count` programs; once over time, stop after the first one."""
        programs = []
        for _ in range(count):
            if programs and self.over_time():
                break
            code = self.generate(env, problem, context)
            if code:
                programs.append(code)
        return programs

    def solve(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        if self.config["clean_contradictions"] and not has_signature(problem["prompt"]):
            problem = self.clean_task(env, problem)
        candidates = self.sample(env, problem, self.config["num_candidates"])
        if not candidates:
            return ""

        probes = None if self.over_time() else self.probes(env, problem)
        outputs = self.run_programs(env, problem, candidates, probes) if probes else {}
        stable = {n: o for n, o in outputs.items() if not nondeterministic(candidates[int(n[1:])])}
        evidence = self.evidence(problem, probes, stable) if stable else ""

        question = self.write_question(env, problem, evidence) if env.can_ask() else None
        answer = None
        if question and env.can_ask():
            try:
                answer = env.ask_human(question)
            except Exception:  # TooManyQuestionException, or a question the harness cannot print
                answer = None

        fallback = [candidates[int(name[1:])] for name in outputs] or candidates
        if self.config["keep_on_non_answer"] and question and says_nothing(answer):
            # nothing was settled, so a program rewritten around the reply is only a fresh guess
            probes = None if self.over_time() else probes
            return self.select_final(env, problem, candidates, [], probes=probes, fallback=fallback)

        context = ANSWER_CONTEXT.format(question=question, answer=answer) if answer else ""
        finals = self.sample(env, problem, self.config["num_final"], context) or candidates[:1]

        tests = []
        if self.config["use_answer_tests"] and question and answer and not self.over_time():
            tests = self.answer_tests(env, problem, question, answer)
        if self.over_time():
            probes = None
        return self.select_final(env, problem, finals, tests, probes=probes, fallback=fallback)
