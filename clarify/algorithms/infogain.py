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
cannot hold together with the rest are dropped first. Tasks with a def line keep
their text as given.

A reply that settles nothing ("not specified", "cannot answer") is not used to
rewrite the program; the programs from before the question are kept instead.

The question reads back what the leading program does: its arguments, its
result, its rule and, when a probe call ran cleanly, one ordinary call with the
output it gave, and asks whether that is what the user wants. When an example in the task disagrees with
the text, the question asks about that example instead. If no read-back comes
out, a task without a def line is asked how the function is called.

Tasks that need files, the network or the shell may import anything; our own
runs then block the network and evalplus. Tasks with DataFrames, arrays, dates
or a class get probes that build such objects, and results with no literal form
are read back by type and shape. A plain yes puts the read-back program first.
Positional calls with too many or too few arguments are fitted to the function
(CALL_ADAPTER, see issue #46). The lower-case spelling of the name is bound too
when it is free (issue #7).
The smaller settings are listed in DEFAULT_CONFIG.

Team: RankOne
Team Members: Md Zahidul Haque, Saima Afrin, Antonio Mastropaolo
Main Contact: mhaque@wm.edu
"""

from __future__ import annotations

import ast
import builtins
import json
import keyword
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
{rules}Put the code in ```python and ```.
""".strip()

NO_IO_SENTENCE = " Do not access files, the network or the shell, and do not use eval or exec."

CODE_RULES = """Make it return a result for every input of the kind the task describes: do not add checks that
raise errors unless the task asks for them, and do not assume the input is sorted, distinct or
non-empty unless the task says so. Accept a list or a tuple wherever a sequence is passed. Where the
task is vague, do what the name `{entry_point}` suggests.
"""

PROBE_TEMPLATE = """
Task (implement `{entry_point}`):
{prompt}

Write {num_probes} diverse argument lists for calling `{entry_point}` that would expose differences
between plausible interpretations of the task (edge cases, boundary values, unusual formats).
Return ONE Python literal: a list where each element is a list of positional arguments.
Enclose it in ```python and ```. Use only literals (no function calls, no imports).
""".strip()

PROBE_LITERALS = """Return ONE Python literal: a list where each element is a list of positional arguments.
Enclose it in ```python and ```. Use only literals (no function calls, no imports)."""

PROBE_EXPRESSIONS = """Return ONE Python list where each element is a list of positional arguments.
Enclose it in ```python and ```. Use literals; an argument that has no literal form (a DataFrame, an
array, a date, an object of a class from the task) may be one constructor call such as
pd.DataFrame({...}) or np.array([...]). No other calls, no lambdas, no imports."""

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

SKIP_ERRORS = """skip input validation, error handling and
performance: the tests use ordinary valid inputs."""

ASK_ERRORS = """skip performance. The task mentions errors, so
which inputs raise and what is raised can be the open point if the text leaves it unclear."""

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

READBACK_TEMPLATE = """
Task (the tests call `{entry_point}`):
{prompt}

Our current program:
```python
{draft}
```
{evidence}
Before writing the final version we will ask the user one question about this program.

If an example in the task disagrees with the task's own text, reply with one line
CLASH: <the question>
that quotes the example call with its stated result, says what the text implies for that same call,
and asks which of the two is intended.

Otherwise describe what the program above does on ordinary inputs, in plain words without code, in
three lines:
ARGS: <the arguments `{entry_point}` takes, in order, with their types and shape>
RETURNS: <the exact type and format of the result>
RULE: <what it computes, in one sentence and in the task's own terms; where the drafts disagreed
(notes above), say what this program does there>
Keep each line under 20 words, and do not name libraries or helper functions.
""".strip()

RULE_TAIL = "say what this program does there>"

RULE_ERRORS = "say what this program does there; if it raises an error, say on which inputs>"

READBACK_NOTES = """
Notes (the user has not seen these): drafts of the program were run on the same inputs and gave
different results here:
{lines}
"""

DRAFT_LIMIT = 4000

ANSWER_CONTEXT = """
The user was asked: {question}
They answered: {answer}
The answer is the final word on that point. If it disagrees with the task text, go with the answer;
everything else follows the task.
"""

CONTRACT_NOTE = """
The user's answer fixes the call: `{entry_point}` takes {count} argument(s). Define it with exactly
that many parameters.
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

# expression probes are built in each program's namespace, fresh for every call
EXPR_RUNNER = (
    RUNNER.replace(
        "_ENTRY = {entry!r}\n",
        """_ENTRY = {entry!r}
_MODULES = {modules}
_base = {{}}
for _alias, _module in _MODULES.items():
    try:
        import importlib
        _base[_alias] = importlib.import_module(_module)
    except BaseException:
        pass
_spaces = {{}}


def _make(args, name):
    if isinstance(args, str):
        return eval(args, {{**_spaces[name], **_base, "nan": float("nan"), "inf": float("inf")}})
    return copy.deepcopy(args)


""",
    )
    .replace(
        "        _fns[_name] = _ns[_ENTRY]\n",
        "        _fns[_name] = _ns[_ENTRY]\n        _spaces[_name] = _ns\n",
    )
    .replace("_fn(*copy.deepcopy(_args))", "_fn(*_make(_args, _name))")
)

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

# block network and evalplus on open-code tasks
CALL_ADAPTER = """


def _infogain_fit_calls(fn, name):
    import builtins as b
    import inspect

    if not inspect.isfunction(fn):
        return fn
    try:
        params = b.list(inspect.signature(fn, follow_wrapped=False).parameters.values())
    except Exception:
        return fn
    if not params or b.any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
        return fn
    most = b.len(params)
    least = b.sum(1 for p in params if p.default is p.empty)
    scope = fn.__globals__

    def fit(args):
        args = b.list(args)
        while b.len(args) > most:
            sizes = b.set()
            for a in args:
                if not b.isinstance(a, (b.bool, b.int)):
                    try:
                        sizes.add(b.len(a))
                    except Exception:
                        pass
            for i, a in b.enumerate(args):
                if b.type(a) is b.int and a in sizes:
                    del args[i]
                    break
            else:
                if most == 1:
                    return [b.tuple(args)]
                del args[most:]
        return args

    def grow(args):
        if b.len(args) == 1 and b.isinstance(args[0], (b.list, b.tuple)):
            if least <= b.len(args[0]) <= most:
                return b.list(args[0])
        sizes = [b.len(a) for a in args if b.isinstance(a, (b.list, b.tuple, b.str, b.dict, b.set))]
        if sizes:
            return b.list(args) + [sizes[0]] * (least - b.len(args))
        return args

    def call(*args, **kwargs):
        if not kwargs and b.len(args) > most:
            args = fit(args)
        elif not kwargs and b.len(args) < least:
            args = grow(args)
        if scope.get(name) is not call:
            return fn(*args, **kwargs)
        scope[name] = fn  # recursion goes straight to fn
        try:
            return fn(*args, **kwargs)
        finally:
            if scope.get(name) is fn:
                scope[name] = call

    call.__name__, call.__doc__, call.__wrapped__ = fn.__name__, fn.__doc__, fn
    return call


try:
    ENTRY = _infogain_fit_calls(ENTRY, "ENTRY")
except Exception:
    pass
del _infogain_fit_calls
"""

GUARD = """import socket


def _no_network(*args, **kwargs):
    raise OSError("network access is blocked")


socket.create_connection = socket.getaddrinfo = _no_network
socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = _no_network
sys.modules["evalplus"] = None


"""


def died(results) -> bool:
    """True if the batch run returned nothing usable."""
    if results is None:
        return True
    lists = [v for v in results.values() if isinstance(v, list)]
    return bool(lists) and not any(lists)


def guarded(template: str) -> str:
    return template.replace(
        "class _Timeout(BaseException):", GUARD + "class _Timeout(BaseException):"
    )


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
ALWAYS_BANNED = {"evalplus"}


def unsafe_reason(code: str, modules=UNSAFE_MODULES, calls=UNSAFE_CALLS) -> str | None:
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
            if name.split(".")[0] in modules:
                return f"imports {name}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in calls:
                return f"calls {node.func.id}()"
    return None


IO_ACTION = (
    r"\b(?:read|reads|reading|open|opens|opened|opening|save|saves|saved|saving|load|loads|loaded"
    r"|loading|download|downloads|downloaded|downloading|fetch|fetches|fetched|fetching|request"
    r"|requests|requested|requesting|connect|connects|connected|connecting|walk|walks|walking|delete"
    r"|deletes|deleted|deleting|create|creates|created|creating)\b"
    r"|\b(?:write|writes|writing)\b(?!\s+(?:an?\s+)?(?:python\s+)?(?:function|program|method|class))"
    r"|(?<!\ba\s)(?<!\bthe\s)\b(?:list|lists|listing)\b"
)
IO_OBJECT = (
    r"\b(?:files?|file\s*paths?|director(?:y|ies)|folders?|urls?|https?|web\s*pages?|network"
    r"|sockets?|environment)\b"
)
IO_MODULES = {"os", "subprocess", "requests", "urllib", "shutil", "socket", "pathlib"}
OPEN_CODE = re.compile(
    rf"(?:{IO_ACTION})[^.\n]{{0,25}}?{IO_OBJECT}"
    r"|\b(?:os|subprocess|requests|urllib|shutil|socket|pathlib)\.\w"
    r"|\bshell\s+(?:commands?|scripts?)\b"
    r"|\b(?:eval|exec)\s*\(|\b(?:use|using|call|calling)\s+(?:the\s+)?(?:built-?in\s+)?(?:eval|exec)\b",
    flags=re.IGNORECASE,
)
NAMED_MODULES = r"^\s*(?:import|from)\s+([\w.]+)|^\s*[-*]\s*([\w.]+)\s*$"


def needs_any_code(prompt: str) -> bool:
    """Check if the task needs files, network, shell or eval."""
    text = prompt or ""
    if OPEN_CODE.search(text):
        return True
    named = re.findall(NAMED_MODULES, text, flags=re.MULTILINE)
    return any(n.split(".")[0] in IO_MODULES for group in named for n in group if n)


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


def says_nothing(answer: str | None, skip_brackets: bool = False) -> bool:
    text = answer or ""
    if skip_brackets:
        # a bracketed aside does not void the answer; all brackets is kept whole
        outside = unbracket(text)
        text = outside if re.search(r"\w", outside) else text
    return bool(re.search(NON_ANSWER, text, flags=re.IGNORECASE))


def unbracket(text: str) -> str:
    while True:
        stripped = re.sub(r"\([^()]*\)", " ", text)
        if stripped == text:
            return text
        text = stripped


ASK_ONLY = (
    r"(?:(?:sorry|pardon(?: me)?|excuse me)[,.!]?\s*)?"
    r"(?:what do you mean(?: by [^?]{1,60})?|which one(?: do you mean)?"
    r"|(?:could|can) you (?:please )?(?:clarify|rephrase|explain)(?: [^?]{1,60})?"
    r"|sorry|pardon(?: me)?)\s*\?"
)


def asks_back(answer: str | None) -> bool:
    """Check if the reply only asks to clarify."""
    text = " ".join((answer or "").split()).strip("\"'*` ")
    return (
        bool(text) and re.fullmatch(rf"(?:{ASK_ONLY}\s*)+", text, flags=re.IGNORECASE) is not None
    )


CONFIRM = r"^\W*(?:yes|correct|that'?s (?:right|correct)|that is (?:right|correct))\b"

CORRECTION = (
    r"\b(?:but|however|instead|should|must|rather|not|except|unless|although|though|also|plus"
    r"|additionally|as\s+well\s+as|and\s+returns?)\b|n't\b"
)


def confirms(answer: str | None) -> bool:
    """Check if the reply is a plain yes."""
    text = (answer or "").replace("\u2019", "'")
    return bool(
        re.match(CONFIRM, text, flags=re.IGNORECASE)
        and not re.search(CORRECTION, text, flags=re.IGNORECASE)
    )


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


FIELD = r"^[\s*>`\-\d.)]*(ARGS|RETURNS|RULE|CLASH)[\s*]*:[\s*]*(.*)$"

CALL = r"\w+\((?:[^()]|\([^()]*\))*\)"


def fill(template: str, **fields: str) -> str:
    """Fill {name} slots in one pass, so braces inside the task or the program stay as they are."""
    return re.sub(r"\{(\w+)\}", lambda m: fields.get(m.group(1), m.group(0)), template)


def plain(text: str, size: int = 300) -> str:
    """One clause for the read-back: no code marks, no question marks, no closing full stop."""
    text = " ".join(text.replace("`", "").replace("?", "").replace("*", "").split()).rstrip(" .;")
    return clip(text, size) if text else ""


def parse_fields(text: str) -> dict[str, str]:
    """The marked lines of the read-back reply; a wrapped sentence is joined back to its line."""
    raw, last = {}, None
    for line in (text or "").splitlines():
        line = line.strip()
        match = re.match(FIELD, line, flags=re.IGNORECASE)
        if match:
            last = match.group(1).upper()
            if last in raw:
                last = None  # a repeated field: keep the first one
            else:
                raw[last] = match.group(2).strip()
        elif last and line[:1].islower() and not raw[last].endswith((".", "!", "?")):
            raw[last] += " " + line
        else:
            last = None
    return {k: v.strip() if k == "CLASH" else plain(v) for k, v in raw.items()}


def cites_task_call(question: str, prompt: str) -> bool:
    """A clash question has to quote a call that is really written in the task."""
    key = lambda s: re.sub(r"\s+", "", squash(s))  # noqa: E731
    return any(key(call) in key(prompt) for call in re.findall(CALL, question))


def edge_input(args) -> bool:
    """An empty, zero, None or negative argument: the tests rarely hinge on such a call."""
    for a in args:
        if a is None or (isinstance(a, int | float) and not isinstance(a, bool) and a <= 0):
            return True
        if isinstance(a, str | list | tuple | dict | set) and len(a) == 0:
            return True
    return not args


def render_readback(entry_point: str, args: str, returns: str, rule: str, example) -> str:
    text = f"About {entry_point}: my current version takes {args} and returns {returns}; {rule}."
    if example:
        text += f" For example, {example[0]} returns {example[1]}."
    return text + READBACK_CLOSE


READBACK_CLOSE = (
    " Is that what you intend, or should it behave differently or return a different format?"
)


def has_signature(prompt: str) -> bool:
    return re.search(r"\bdef\s+\w+\s*\(", prompt or "") is not None


def contract_known(prompt: str, entry_point: str) -> bool:
    """Check if the task shows how the entry point is called."""
    name = re.escape(entry_point)
    shown = (
        r"\bdef\s+\w+\s*\(",
        rf"^\s*class\s+{name}\s*[:(]",
        rf"\b(?i:class\s+(?:named|called))\s+[`'\"]?{name}\b",
        rf"\b{name}\(\s*[^\s)]",
    )
    return any(re.search(p, prompt or "", flags=re.MULTILINE) for p in shown)


ERRORS = (
    r"\b\w+(?:Error|Exception)\b"
    r"|(?i:\brais(?:e|es|ed)\s+an?\b(?![^.\n]*\bpower\b)|\bthrows?\b|\ban\s+exception\b)"
)


def mentions_errors(prompt: str) -> bool:
    """Check if the task mentions errors."""
    return re.search(ERRORS, prompt or "") is not None


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


# type summaries


def is_literal(value: str) -> bool:
    try:
        ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return False
    return True


def loose_literal(text: str, loose: bool = True):
    """literal_eval with nan and inf as None (unless loose is False)."""
    if loose:
        text = re.sub(r"(?<![\w'\"])-?(?:nan|inf)\b", "None", text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def array_shape(items) -> tuple:
    if not isinstance(items, list):
        return ()
    inner = {array_shape(x) for x in items}
    return (len(items), *inner.pop()) if len(inner) == 1 else (len(items),)


def leaves(items):
    if isinstance(items, list):
        for x in items:
            yield from leaves(x)
    else:
        yield items


KINDS = {bool: "booleans", int: "integers", float: "floats", type(None): "floats", str: "strings"}


def array_summary(name: str, payload: str) -> str:
    noun = "an array" if name == "ndarray" else article(name)
    items = loose_literal(payload)
    if not isinstance(items, list):
        return noun
    kinds = {KINDS.get(type(x)) for x in leaves(items)}
    kind = f" of {next(iter(kinds))}" if len(kinds) == 1 and None not in kinds else ""
    return f"{noun} of shape {array_shape(items)}{kind}"


def frame_summary(name: str, payload: str) -> str:
    text, parts = article(name), loose_literal(payload)
    if not isinstance(parts, dict):
        return text
    index, columns = parts.get("index"), parts.get("columns")
    rows = f" with {plural(len(index), 'row')}" if isinstance(index, list) else ""
    if not isinstance(columns, list):
        return text + rows
    joint = " and" if rows else " with"
    full = f"{text}{rows}{joint} columns {columns!r}"
    return full if len(full) <= 80 else f"{text}{rows}{joint} {plural(len(columns), 'column')}"


def object_summary(name: str, state: str | None) -> str:
    fields = loose_literal(state, loose=False) if state else None
    if not isinstance(fields, dict):
        return article(name) + " object"
    # public attributes only
    pairs = [f"{k}={v!r}" for k, v in fields.items() if isinstance(k, str) and k[:1] != "_"]
    return f"{article(name)} with {', '.join(pairs)}" if pairs else article(name) + " object"


def counted(prefix: str, payload: str, word: str) -> str:
    items = loose_literal(payload)
    return f"{prefix} of {plural(len(items), word)}" if isinstance(items, list) else prefix


def type_summary(value: str) -> str | None:
    """Describe a non-literal result by type and shape."""
    if value in ("nan", "inf", "-inf"):
        return value
    flags = re.DOTALL
    if match := re.fullmatch(r"([A-Za-z_]\w*):(.*)", value, flags=flags):
        name, payload = match.groups()
        if payload.startswith("{"):
            text = frame_summary(name, payload)
        elif name == "Series":
            text = counted("a Series", payload, "value")
        elif payload.startswith("["):
            text = array_summary(name, payload)
        else:
            return None
    elif match := re.fullmatch(r"(set|frozenset)\{(.*)\}", value, flags=flags):
        text = counted(f"a {match.group(1)}", f"[{match.group(2)}]", "item")
    elif match := re.fullmatch(r"iter(\[.*\])", value, flags=flags):
        text = counted("an iterator", match.group(1), "item")
    elif match := re.fullmatch(r"([A-Za-z_][\w.]*)(?:\(.*\)|(\{.*\}))?", value, flags=flags):
        text = object_summary(match.group(1).rsplit(".", 1)[-1], match.group(2))
    else:
        return None
    return clip(text, 80)


# expression probes


PROBE_MODULES = {
    "pd": "pandas",
    "pandas": "pandas",
    "np": "numpy",
    "numpy": "numpy",
    "datetime": "datetime",
    "collections": "collections",
    "decimal": "decimal",
    "fractions": "fractions",
    "math": "math",
}
PROBE_NODES = (
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
)
REFLECTION = {"sys", "getattr", "setattr", "delattr", "globals", "locals", "vars"}
PROBE_BLOCKED = UNSAFE_CALLS | UNSAFE_MODULES | REFLECTION
ARRAY_MODULES = {"pd", "pandas", "np", "numpy"}
CONSTRUCTORS = {
    "DataFrame",
    "Series",
    "array",
    "arange",
    "zeros",
    "ones",
    "full",
    "to_datetime",
    "Timestamp",
    "date_range",
    "NaT",
    "nan",
    "inf",
}
TYPED_TASK = (
    r"(?i:\bpandas\b|\bnumpy\b|\bdataframes?\b|\bndarrays?\b|\bnp\.\w|\bpd\.\w|\bdtype\b"
    r"|\btensors?\b|\bdatetime\b|\btimestamps?\b|\bdate\s+objects?\b)|\bSeries\b"
)


def typed_task(prompt: str, entry_point: str) -> bool:
    """Check if the task uses DataFrames, arrays, dates or a class."""
    text = prompt or ""
    if re.search(TYPED_TASK, text):
        return True
    return re.search(rf"^\s*class\s+{re.escape(entry_point)}\s*[:(]", text, flags=re.M) is not None


def safe_probe(node: ast.AST) -> bool:
    """Allow only literals and constructor calls."""
    for n in ast.walk(node):
        if not isinstance(n, PROBE_NODES):
            return False
        if isinstance(n, ast.Name) and (n.id.startswith("__") or n.id in PROBE_BLOCKED):
            return False
        if isinstance(n, ast.Attribute) and (
            n.attr.startswith("__") or not isinstance(n.value, ast.Name)
        ):
            return False
        if isinstance(n, ast.Attribute) and n.value.id in ARRAY_MODULES:
            if n.attr not in CONSTRUCTORS:
                return False
        if isinstance(n, ast.keyword) and (n.arg is None or n.arg.startswith("__")):
            return False
        if isinstance(n, ast.Dict) and None in n.keys:
            return False
    return True


def expression_probes(block: str) -> list | None:
    """Parse a probe list that has constructor calls."""
    try:
        tree = ast.parse(block.strip(), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.List):
        return None
    probes = []
    for node in tree.body.elts:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            if safe_probe(node):
                args = node.elts if isinstance(node, ast.List | ast.Tuple) else [node]
                probes.append("[" + ", ".join(ast.unparse(a) for a in args) + "]")
            continue
        probes.append(value if isinstance(value, list | tuple) else [value])
    return probes or None


def probe_runner(probes) -> tuple[str, dict[str, str]]:
    """Pick the runner for literal or expression probes."""
    sources = [p for p in probes if isinstance(p, str)]
    if not sources:
        return RUNNER, {}
    trees = [ast.parse(p, mode="eval") for p in sources]
    names = {n.id for t in trees for n in ast.walk(t) if isinstance(n, ast.Name)}
    return EXPR_RUNNER, {"modules": repr({k: v for k, v in PROBE_MODULES.items() if k in names})}


def edge_probe(probe) -> bool:
    """edge_input for expression probes too."""
    if not isinstance(probe, str):
        return edge_input(probe)
    nodes = ast.parse(probe, mode="eval").body.elts
    values = []
    for node in nodes:
        if isinstance(node, ast.Call) and not node.args and not node.keywords:
            return True  # empty constructor, e.g. pd.DataFrame()
        inner = (
            [*node.args, *(k.value for k in node.keywords)]
            if isinstance(node, ast.Call)
            else [node]
        )
        for n in inner:
            try:
                values.append(ast.literal_eval(n))
            except (ValueError, TypeError, SyntaxError):
                continue
    return not nodes or (bool(values) and edge_input(values))


def call_text(entry_point: str, probe) -> str:
    if isinstance(probe, str):
        return f"{entry_point}({probe[1:-1]})"
    return f"{entry_point}({', '.join(repr(a) for a in probe)})"


# arity from the answer

NUMBERS = {
    "no": 0,
    "zero": 0,
    "one": 1,
    "a single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
COUNT = r"\d+|no|zero|one|a\s+single|two|three|four|five|six|seven|eight|nine|ten"


def count_of(word: str) -> int | None:
    word = " ".join(word.lower().split())
    if word.isdigit():
        return int(word) if len(word) <= 2 else None
    return NUMBERS[word]


def call_arity(inner: str) -> int:
    if not inner.strip():
        return 0
    try:
        call = ast.parse(f"_({inner})", mode="eval").body
        return len(call.args) + len(call.keywords)
    except SyntaxError:
        depth, count = 0, 1
        for ch in inner:
            depth += ch in "([{"
            depth -= ch in ")]}"
            count += ch == "," and depth == 0
        return count


def answer_arity(answer: str | None, entry_point: str) -> int | None:
    """Get the argument count the answer states, or None."""
    text = " ".join((answer or "").split())
    name = re.escape(entry_point)
    stated = (
        rf"\b(?:it|the function|{name})\s+(?:(?:should|must|only|now|also)\s+)?takes?\s+"
        rf"(?:(?:exactly|only|just)\s+)?({COUNT})\s+(?:arguments?|parameters?|inputs?|values?)\b"
    )
    found = {count_of(m.group(1)) for m in re.finditer(stated, text, flags=re.IGNORECASE)}
    for m in re.finditer(rf"\b{name}\(((?:[^()]|\([^()]*\))*)\)", text):
        found.add(call_arity(m.group(1)))
    found.discard(None)
    return found.pop() if len(found) == 1 else None


def method(cls: ast.AST | None, name: str):
    if not isinstance(cls, ast.ClassDef):
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def param_range(code: str, entry_point: str) -> tuple[int, float] | None:
    """Return (min, max) positional args of entry_point, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    kinds = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    defs = {n.name: n for n in tree.body if isinstance(n, kinds)}
    node, bound = defs.get(entry_point), False
    for n in tree.body:
        # alias added by bind_entry_point
        if node is None and isinstance(n, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == entry_point for t in n.targets):
                v = n.value
                if isinstance(v, ast.Name):
                    node = defs.get(v.id)
                elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Call):
                    if isinstance(v.value.func, ast.Name):
                        node, bound = method(defs.get(v.value.func.id), v.attr), True
    if isinstance(node, ast.ClassDef):
        node, bound = method(node, "__init__"), True
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list)
    positional = [*node.args.posonlyargs, *node.args.args][1 if bound and not static else 0 :]
    most = math.inf if node.args.vararg else len(positional)
    return max(len(positional) - len(node.args.defaults), 0), most


def takes(code: str, entry_point: str, count: int) -> bool:
    span = param_range(code, entry_point)
    return span is not None and span[0] <= count <= span[1]


class InfoGainClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "num_candidates": 3,  # programs sampled before asking
        "num_probes": 8,  # inputs used to compare them
        "max_evidence": 3,  # disagreeing inputs shown to the question writer
        "num_final": 2,  # programs sampled after the answer
        "use_answer_tests": True,  # pick the final program with asserts from the answer
        "clean_contradictions": True,  # no def line: drop statements that clash with the rest first
        "call_note": True,  # no def line, no read-back: ask how the function is called
        "witness": False,  # ask with a concrete call and two possible results
        "evidence_quote": False,  # the question must point at words that are really in the task
        "keep_on_non_answer": True,  # a reply that settles nothing does not replace the programs we have
        "readback_question": True,  # read the leading program back and ask if it is meant
        "readback_without_def": True,  # no def line: the read-back too, instead of the call question
        "coding_rules": True,  # total functions, no assumed preconditions
        "time_limit": 200,  # seconds; past this, skip the optional steps (harness stops at 300)
        "allow_any_code": True,  # file/network/shell tasks may import anything but evalplus
        "answer_first_fallback": True,  # keep the final if the drafts fail too
        "isolate_on_failure": True,  # rerun one program at a time if the batch dies
        "typed_values": True,  # read back a DataFrame, array, set or object by type and shape
        "expression_probes": True,  # DataFrame/array/date/class tasks: probes may construct objects
        "contract_routing": True,  # a call of the entry point in the task counts as a signature
        "error_aware": True,  # ask about errors when the task names them
        "bind_answer_contract": False,  # "it takes N arguments" in the answer ranks the finals
        "confirmed_draft": True,  # trust a plain yes, ignore a counter-question
        "non_answer_brackets": True,  # ignore "(not specified)" asides
        "call_adapter": True,  # fit mismatched calls, also bind a lower-case name
    }

    # per-task state, reset in run(); the repo validator rejects underscore attributes
    kept = ()  # programs that came back clean during the current task
    deadline = math.inf
    read_draft = None  # the program the question read back
    open_code = False  # file/network/shell task
    spare_runs = 0  # one-by-one reruns left

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

    def contract(self, problem):
        if self.config["contract_routing"]:
            return contract_known(problem["prompt"], problem["entry_point"])
        return has_signature(problem["prompt"])

    def refusal(self, code):
        """Reason to refuse the program, or None."""
        if not self.config["allow_any_code"]:
            return unsafe_reason(code)
        if self.open_code:
            return unsafe_reason(code, ALWAYS_BANNED, set())
        return unsafe_reason(code, UNSAFE_MODULES | ALWAYS_BANNED)

    def generate(self, env, problem, context=""):
        template = CODE_TEMPLATE.replace(NO_IO_SENTENCE, "") if self.open_code else CODE_TEMPLATE
        prompt = (
            template.replace("{rules}", CODE_RULES if self.config["coding_rules"] else "")
            .replace("{entry_point}", problem["entry_point"])
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
                reason = self.refusal(code)
                if reason is None:
                    self.kept = [*self.kept, code]
                    return code
                feedback = (
                    f"The code {reason}. Rewrite it as a plain function without files, network, "
                    "shell, eval or exec. Put the code in ```python and ```."
                )
                if self.open_code:
                    feedback = f"The code {reason}, which is not available. Rewrite it without it. Put the code in ```python and ```."
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": feedback},
            ]
        return None

    def probes(self, env, problem):
        expressions = self.config["expression_probes"] and typed_task(
            problem["prompt"], problem["entry_point"]
        )
        template = PROBE_TEMPLATE
        if expressions:
            template = template.replace(PROBE_LITERALS, PROBE_EXPRESSIONS)
        response = self.chat(
            env,
            template.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{num_probes}", str(self.config["num_probes"])),
        )
        block = extract_block(response, "python")
        if not block:
            return None
        try:
            probes = ast.literal_eval(block.strip())
        except (ValueError, SyntaxError):
            return expression_probes(block) if expressions else None
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

    def sandbox(self, template):
        """Add the network guard on open-code tasks."""
        return guarded(template) if self.open_code else template

    def batch(self, env, problem, programs, probes):
        runner, extra = probe_runner(probes)
        return self.execute(
            env,
            self.sandbox(runner),
            sources=repr({f"c{i}": c for i, c in enumerate(programs)}),
            probes=repr(probes),
            entry=problem["entry_point"],
            deadline=EXEC_DEADLINE_SECONDS,
            **extra,
        )

    def probe_results(self, env, problem, programs, probes):
        """Run the probes, one program at a time if the batch dies."""
        results = self.batch(env, problem, programs, probes)
        if not self.config["isolate_on_failure"] or len(programs) < 2 or not died(results):
            return results
        budget, self.spare_runs = self.spare_runs, 0  # one round per task
        merged = {}
        for i, program in enumerate(programs):
            if budget <= 0 or self.over_time():
                break
            budget -= 1
            alone = self.batch(env, problem, [program], probes)
            if alone and "c0" in alone:
                merged[f"c{i}"] = alone["c0"]
        return merged or results

    def run_programs(self, env, problem, candidates, probes):
        """Outputs per program on the probes; programs that fail to load are left out."""
        results = self.probe_results(env, problem, candidates, probes)
        if results is None:
            return {}
        return {name: outs for name, outs in results.items() if isinstance(outs, list) and outs}

    # question

    def evidence(self, problem, probes, outputs):
        lines = []
        for j in split_probes(outputs, self.config["max_evidence"]):
            call = clip(call_text(problem["entry_point"], probes[j]))
            groups = Counter(outs[j] for outs in outputs.values())
            shown = "; ".join(f"{n} program(s) return {clip(v, 80)}" for v, n in groups.items())
            lines.append(f"- {call}: {shown}")
        return "\n".join(lines)

    def errors_open(self, problem):
        return self.config["error_aware"] and mentions_errors(problem["prompt"])

    def write_question(self, env, problem, evidence):
        open_call = not self.contract(problem)
        call = CALL_NOTE if open_call and self.config["call_note"] else ""
        hint = HINT.replace("{lines}", evidence) if evidence else ""
        style = WITNESS_STYLE if self.config["witness"] else PLAIN_STYLE
        quote = ""
        if self.config["evidence_quote"]:
            quote = QUOTE_NOTE.replace("{none}", QUOTE_NONE if open_call else "")
        template = QUESTION_TEMPLATE
        if self.errors_open(problem):
            template = template.replace(SKIP_ERRORS, ASK_ERRORS)
        prompt = (
            template.replace("{prompt}", problem["prompt"])
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

    @staticmethod
    def leading(candidates, outputs):
        """The program that crashes least on the probes, then the one most others agree with."""
        groups = {}
        for name in sorted(outputs, key=lambda n: int(n[1:])):
            groups.setdefault(repr(outputs[name]), []).append(name)
        if not groups:
            return None, candidates[0]

        def rank(group):
            crashes = sum(str(v).startswith("ERR:") for v in outputs[group[0]])
            return -crashes, len(group)

        name = max(groups.values(), key=rank)[0]
        return name, candidates[int(name[1:])]

    @staticmethod
    def example_call(problem, probes, outputs, name, typed=False):
        """An ordinary probe call and the output the program really gave, where the drafts split first."""
        if not name or not probes:
            return None
        outs = outputs[name]
        split = split_probes(outputs, len(outs))
        order = split + [j for j in range(len(outs)) if j not in split]
        order.sort(key=lambda j: edge_probe(probes[j]))  # stable: ordinary calls first
        calls = [(call_text(problem["entry_point"], probes[j]), str(outs[j])) for j in order]
        calls = [(call, value) for call, value in calls if len(call) <= 120]
        for call, value in calls:
            if len(value) <= 80 and is_literal(value):
                return call, value
        # else a type summary
        for call, value in calls if typed else ():
            summary = type_summary(value)
            if summary:
                return call, summary
        return None

    def readback_question(self, env, problem, candidates, probes, outputs, evidence):
        """What the leading program does, read back with a real call; None to fall back."""
        name, draft = self.leading(candidates, outputs)
        if len(draft) > DRAFT_LIMIT:
            return None
        template = READBACK_TEMPLATE
        if self.errors_open(problem):
            template = template.replace(RULE_TAIL, RULE_ERRORS)
        prompt = fill(
            template,
            prompt=problem["prompt"],
            draft=draft,
            evidence=READBACK_NOTES.replace("{lines}", evidence) if evidence else "",
            entry_point=problem["entry_point"],
        )
        fields = parse_fields(self.chat(env, prompt))
        if fields.get("CLASH"):
            question = tidy_question(fields["CLASH"])
            if question and cites_task_call(question, problem["prompt"]):
                return question
        parts = [fields.get(k) for k in ("ARGS", "RETURNS", "RULE")]
        if not all(parts):
            return None
        example = self.example_call(problem, probes, outputs, name, self.config["typed_values"])
        self.read_draft = draft
        return render_readback(problem["entry_point"], *parts, example)

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
        results = self.probe_results(env, problem, programs, probes)
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

    def last_resort(self, env, problem, finals, probes, fallback, check_drafts):
        """Fall back to the drafts unless they fail too."""
        if not (check_drafts and self.config["answer_first_fallback"]):
            return fallback[0]
        runs = self.healthy(env, problem, fallback, probes)
        return finals[0] if runs == [] else fallback[0]

    def select_final(
        self,
        env,
        problem,
        finals,
        tests,
        probes=None,
        fallback=None,
        arity=None,
        check_drafts=False,
        by_order=False,
    ):
        if probes:
            ok = self.healthy(env, problem, finals, probes)
            if ok == [] and fallback:
                return self.last_resort(env, problem, finals, probes, fallback, check_drafts)
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
                    self.sandbox(TEST_RUNNER),
                    sources=repr(sources),
                    tests=repr(tests),
                    deadline=EXEC_DEADLINE_SECONDS,
                )
                or {}
            )
        best = max(scores.values(), default=0)
        pool = [c for i, c in enumerate(finals) if scores.get(f"c{i}", 0) == best] or finals
        if arity is not None:
            # prefer the arity the user gave
            pool = [c for c in pool if takes(c, problem["entry_point"], arity)] or pool
        if by_order:
            return pool[0]  # confirmed draft wins ties
        # tie-break: most common program, else the first
        return Counter(pool).most_common(1)[0][0]

    # main

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        self.kept = []
        self.read_draft = None
        self.open_code = False
        self.spare_runs = 0
        self.deadline = time.monotonic() + self.config["time_limit"]
        try:
            code = self.solve(env, problem)
        except LimitsExceededException:
            # out of model budget: the last program that came back clean beats nothing
            code = self.kept[-1] if self.kept else ""
        except Exception:
            # anything else (an api error, a reply the harness cannot print): same fallback
            code = self.kept[-1] if self.kept else ""
        return self.fit_calls(code, problem)

    def fit_calls(self, code, problem):
        """Append CALL_ADAPTER and a lower-case alias of the entry point to a program."""
        if not self.config["call_adapter"]:
            return code
        try:
            entry = problem.get("entry_point")
            if not (
                isinstance(entry, str) and entry.isidentifier() and not keyword.iskeyword(entry)
            ):
                return code
            if not (code and code.strip()) or "```" in code:
                return code
            fitted = code.rstrip() + CALL_ADAPTER.replace("ENTRY", entry)
            lower = entry.lower()
            if lower != entry and not keyword.iskeyword(lower) and not hasattr(builtins, lower):
                # some tasks give the name with capitals (issue #7)
                fitted += f'\nif "{lower}" not in globals() and "{entry}" in globals():\n    {lower} = {entry}\n'
            ast.parse(code)
            ast.parse(fitted)
        except Exception:
            return code
        return fitted

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

    def unsettled(self, answer):
        """Check if the reply settles nothing."""
        if says_nothing(answer, self.config["non_answer_brackets"]):
            return True
        return self.config["confirmed_draft"] and asks_back(answer)

    def bind_contract(self, env, problem, finals, context, answer):
        """Get the stated arity, adding a final that fits it if none does."""
        if not (self.config["bind_answer_contract"] and answer):
            return None, finals
        entry = problem["entry_point"]
        arity = answer_arity(answer, entry)
        if arity is None or any(takes(f, entry, arity) for f in finals) or self.over_time():
            return arity, finals
        note = CONTRACT_NOTE.replace("{entry_point}", entry).replace("{count}", str(arity))
        try:
            extra = self.generate(env, problem, context + note)
        except LimitsExceededException:
            return arity, finals
        return arity, [*finals, extra] if extra else finals

    def solve(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        self.open_code = self.config["allow_any_code"] and needs_any_code(problem["prompt"])
        if self.config["clean_contradictions"] and not has_signature(problem["prompt"]):
            problem = self.clean_task(env, problem)
        candidates = self.sample(env, problem, self.config["num_candidates"])
        if not candidates:
            return ""
        self.spare_runs = len(candidates)

        probes = None if self.over_time() else self.probes(env, problem)
        outputs = self.run_programs(env, problem, candidates, probes) if probes else {}
        stable = {n: o for n, o in outputs.items() if not nondeterministic(candidates[int(n[1:])])}
        evidence = self.evidence(problem, probes, stable) if stable else ""

        question = None
        if env.can_ask():
            readback = self.config["readback_question"] and (
                self.contract(problem) or self.config["readback_without_def"]
            )
            if readback and not self.over_time():
                question = self.readback_question(
                    env, problem, candidates, probes, stable, evidence
                )
            question = question or self.write_question(env, problem, evidence)
        answer = None
        if question and env.can_ask():
            try:
                answer = env.ask_human(question)
            except Exception:  # TooManyQuestionException, or a question the harness cannot print
                answer = None

        fallback = [candidates[int(name[1:])] for name in outputs] or candidates
        if self.config["keep_on_non_answer"] and question and self.unsettled(answer):
            # nothing was settled, so a program rewritten around the reply is only a fresh guess
            probes = None if self.over_time() else probes
            return self.select_final(env, problem, candidates, [], probes=probes, fallback=fallback)

        context = ANSWER_CONTEXT.format(question=question, answer=answer) if answer else ""
        finals = self.sample(env, problem, self.config["num_final"], context) or candidates[:1]
        arity, finals = self.bind_contract(env, problem, finals, context, answer)
        confirmed = bool(self.config["confirmed_draft"] and self.read_draft and confirms(answer))
        if confirmed:
            finals = [self.read_draft, *finals]

        tests = []
        if self.config["use_answer_tests"] and question and answer and not self.over_time():
            tests = self.answer_tests(env, problem, question, answer)
        if self.over_time():
            probes = None
        # test the drafts only if they never ran
        check = bool(answer) and not outputs
        return self.select_final(
            env, problem, finals, tests, probes, fallback, arity, check, by_order=confirmed
        )
