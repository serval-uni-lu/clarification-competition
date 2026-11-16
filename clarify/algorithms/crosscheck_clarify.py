# SPDX-FileCopyrightText: 2026 Khalid Tounoussi <khalid.tounoussi.1@ens.etsmtl.ca>
#
# SPDX-License-Identifier: MIT

"""CrossCheckClarification.

First, a sentence or clause that cannot hold together with the rest of the requirement (an
impossible implementation limit, numbers returned as strings, a closing "but ..." that cancels
the main statement) is deleted, and only a verified pure deletion is kept; the user still sees
the original text, the algorithm's own steps read the cleaned one.
Gathers evidence before asking: three drafts are run on the examples the requirement states
and on probe inputs. The single question states the draft's assumed contract (arguments,
result format, rule, one example) so the user can correct whichever part is wrong; when the
text and a stated example disagree, it asks only which of the two is intended. Three final
implementations are then written with the answer. Stated examples and calls the answer
settles become checks, except a check that no final satisfies (the answer overrides it);
checks select the final, drive repair and gate an answer-compliance pass, and execution
agreement on the probes breaks ties.
With a second clarification turn available (run with --max_clarification_turns 2), one more
question is asked only when the answer corrected the assumed contract (majority of three
classifications). It targets the point the implementation is now most likely to get wrong, and
the final implementations, checks, selection, repair and answer-compliance pass are rerun with
both answers. With one turn the behaviour is exactly the single-question version.

Team: STIL-ETS
Team Members: Khalid Tounoussi, Issam Oukhay
Main Contact: khalid.tounoussi.1@ens.etsmtl.ca
"""

from __future__ import annotations

import ast
import difflib
import io
import json
import re
import threading
import time
import tokenize

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import TooManyQuestionException

RULES = """Rules:
- Implement exactly the function `{entry}` (module level, standard library only). Return the
  result; never only print it. No input(), no tests, no prints, no top-level calls, no
  `if __name__ == "__main__":` block.
- The name `{entry}` was chosen by the user: the behaviour must match what the name says.
- Keep the function total: do not add input validation that raises, and do not take shortcuts
  that are only correct under an assumed precondition (sortedness, distinct or non-empty
  input); compute the answer directly for every input.
- Accept the argument types a caller would naturally pass (lists or tuples, ints or floats)
  without type checks.{signature_rule}"""

SIGNATURE_RULE = """
- No signature is given: use the parameters a typical exercise with this description and
  function name would use, in the order the description introduces them."""

DRAFT_PROMPT = """Implement this requirement in Python.

<requirement>
{prompt}
</requirement>

{rules}

Return only one ```python block."""

FINAL_PROMPT = """Implement this requirement in Python.

<requirement>
{prompt}
</requirement>

You asked the user a clarifying question:
Q: {question}
A: {answer}

The answer is authoritative for what it addresses, even where it contradicts the text or a
stated example. For everything it does not address, follow the requirement text, its stated
examples, and the function name.

{rules}

Return only one ```python block."""

QUESTION_PROMPT = """You are about to deliver an implementation of `{entry}` for this requirement:

<requirement>
{prompt}
</requirement>

Your draft:
```python
{draft}
```

Evidence from running your drafts:
{evidence}

You may ask the user ONE question before delivering.
If the requirement's text and one of its examples disagree, or an example shows something the
text never mentions (an extra prefix, a different result), ask only about that one example and
do not describe your draft: quote the example call with its stated result, state the result the
text implies for that same call, and ask which of the two is intended. Otherwise use the question
to let the user correct your draft's contract where it is most likely wrong. State concretely
what your draft assumes on ordinary inputs: what arguments {entry} takes (their types and shape),
what it returns (exact type and format) and the rule it computes, with one example call and its
output. Then ask whether that is right, or whether it should instead <the most plausible
alternative for the part you are least sure about>, leading with the part the requirement leaves
most open: an unclear result format or argument shape, or the core rule. Do not ask about empty,
zero, negative or invalid input unless nothing else is open. Never ask for code or tests.
Write plain text on one line, without code formatting or backticks. Reply with the question only."""

PHRASINGS = {
    "confirm": 'phrase it as "I will assume <one specific behaviour, with one tiny example input and '
    'its output>. Is that right?"',
    "confirm_alt": 'phrase it as "I will assume <one specific behaviour, with one tiny example input '
    "and its output>. Is that right, or should it <the main alternative, with its output for that "
    'example>?"',
    "neutral": "phrase it as a choice between the two most plausible readings, each with one tiny "
    'example: "Should <name> <reading A, e.g. ...> or <reading B, e.g. ...>?"',
    # Asks for the fact itself, so the user cannot simply agree with a proposed behaviour.
    "open": "phrase it as a narrow open question about that one point, without proposing an answer, "
    'e.g. "What should <name> return for <one tiny example input>?" or "How should <name> treat '
    '<the unclear point>?"',
}


READINGS_PROMPT = """The requirement below may be ambiguous, contradictory or incomplete. List up to {count}
materially different readings of what `{entry}` should do: readings that make it return
DIFFERENT results on ordinary inputs (not only on empty or invalid input). Put the most likely
reading first. Write each as one sentence stating the arguments, what is computed, and the exact
result type and format.

<requirement>
{prompt}
</requirement>

Return only a ```python block defining READINGS, a list of strings."""

READING_DRAFT = """Implement this requirement in Python, under this specific reading of it:
{reading}

<requirement>
{prompt}
</requirement>

{rules}

Return only one ```python block."""

PROBE_PROMPT = """Write {count} small, valid inputs for the Python function `{entry}` described
below: typical cases first, then boundary cases (empty, single element, ties, duplicates,
zero, negative numbers) that the description allows.

<requirement>
{prompt}
</requirement>
{context}
The function is called as `{entry}(*args)`. Signature: {signature}.
Return only a ```python block defining INPUTS, a list where each element is the tuple of
positional arguments of one call, e.g. INPUTS = [([1, 2, 3], 2), ([], 0)]."""

CHECKS_PROMPT = """A requirement, a clarification the user gave, and example calls of `{entry}`.

<requirement>
{prompt}
</requirement>

Q: {question}
A: {answer}

Examples stated in the requirement:
{examples}

1. For each stated example (by number), write CONFIRMED if the clarification explicitly
   supports it, DROP if the clarification contradicts it; otherwise CONFLICT if the example
   contradicts an explicit statement of the requirement text, or KEEP if it is consistent with
   the text (possibly revealing a detail the text leaves out).
2. List concrete example calls the user's answer confirms or states (including the example in
   the question if the answer confirms it), with the result the answer implies. Only calls the
   answer settles; none if it settles no concrete call.

Return only a ```python block defining VERDICTS (list of "CONFIRMED"/"DROP"/"CONFLICT"/"KEEP",
one per stated example) and CONFIRMED (list of (args_tuple, expected) pairs)."""

REPAIR_PROMPT = """Implement this requirement in Python.

<requirement>
{prompt}
</requirement>
{clarification}
Your implementation:
```python
{code}
```

It fails these checks, which come from the requirement's own examples or the user's answer:
{failures}

Find the behaviour that satisfies these checks (an example may reveal a detail the text leaves
implicit) while keeping everything the checks do not concern unchanged. If a check contradicts
the user's answer, follow the answer instead.

{rules}

Return only one ```python block."""

VERIFY_PROMPT = """A user gave this requirement and then answered a clarifying question.

<requirement>
{prompt}
</requirement>

Q: {question}
A: {answer}

Implementation of `{entry}`:
```python
{code}
```

Check the implementation against the user's answer, which overrides the requirement text where
they conflict. If it follows the answer completely, reply with exactly OK. Otherwise reply with a
corrected implementation that follows the answer (change nothing else) in one ```python block."""

QUESTION2_PROMPT = """You are about to deliver an implementation of `{entry}` for this requirement:

<requirement>
{prompt}
</requirement>

You already asked the user one question:
Q: {question}
A: {answer}

Your implementation, which follows that answer:
```python
{code}
```

Evidence from running your implementations:
{evidence}

You may ask the user ONE more question before delivering. It must be about a different point
than the first question: do not repeat, rephrase or narrow it, and do not ask about anything the
answer already settles. Use it to let the user correct your implementation where it is now most
likely wrong, leading with the part the requirement and the answer still leave most open: an
argument shape or result format the answer did not cover, a detail of the core rule, or what a
stated example means. State concretely what your implementation assumes on that one point, with
one example call and its output. Then ask whether that is right, or whether it should instead
<the most plausible alternative>. Do not ask about empty, zero, negative or invalid input unless
nothing else is open. Never ask for code or tests. Write plain text on one line, without code
formatting or backticks. Reply with the question only."""

FINAL2_PROMPT = """Implement this requirement in Python.

<requirement>
{prompt}
</requirement>

You asked the user two clarifying questions:
Q: {question}
A: {answer}
Q: {question2}
A: {answer2}

The answers are authoritative for what they address, even where they contradict the text or a
stated example. For everything they do not address, follow the requirement text, its stated
examples, and the function name.

{rules}

Return only one ```python block."""

# Logged only (the pilot always asks); a candidate gate for when a second question is worth its cost.
GATE_PROMPT = """A requirement for the Python function `{entry}` and a clarifying question the user answered:

<requirement>
{prompt}
</requirement>

Q: {question}
A: {answer}

After this answer, is some point still unclear that would change what `{entry}` returns on
ordinary inputs (not empty, invalid or extreme ones)? Reply with exactly one word: YES or NO."""

# Did the answer confirm, correct or add to the assumption stated in the question?
NEED_PROMPT = """A developer asked a user a clarifying question that states the developer's assumption, and the user answered.

Q: {question}
A: {answer}

Classify the answer with exactly one word:
CONFIRM - the answer agrees with the assumption and adds nothing that changes it
CORRECT - the answer contradicts or changes at least one part of the assumption
ADD - the answer agrees but adds a requirement or detail that the assumption did not state"""

PRECLEAN_PROMPT = """Below is a requirement for the Python function `{entry}`. Some requirements were edited
after they were written and now contain one sentence or clause that cannot hold together with the
rest: an impossible implementation limit (constant time, no loops or no recursion for a task that
must look at every element), a demand to return numbers or booleans as strings, a demand to accept
a different kind of input than the rest describes, or a closing "but ..." / "instead ..." clause
that cancels the main statement.

If the requirement contains such a sentence or clause, reply with the requirement with exactly that
text deleted, inside one ```text block; change nothing else, keep every example, the signature and
the function name, and add no words. Vague wording, missing details, open choices and unusual but
consistent demands are not contradictions; keep them. If there is nothing to delete, reply with the
single word UNCHANGED.

<requirement>
{prompt}
</requirement>"""

ADAPTER = r"""

def _bind_arguments(_fn):
    import copy as _copy
    import functools as _functools
    import inspect as _inspect

    try:
        _signature = _inspect.signature(_fn)
    except (TypeError, ValueError):
        return _fn

    def _fits(args):
        try:
            _signature.bind(*args)
            return True
        except TypeError:
            return False

    def _variants(args):
        params = [p for p in _signature.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        required = [p for p in params if p.default is p.empty]
        sized = (list, tuple, str, dict, set)
        # C-style calls pass each sequence followed by its length: f(a, len(a), b, len(b)).
        companions = [i for i in range(1, len(args)) if isinstance(args[i], int) and not isinstance(args[i], bool)
                      and isinstance(args[i - 1], sized) and args[i] == len(args[i - 1])]
        if companions and len(args) - len(companions) >= len(required):
            yield tuple(a for i, a in enumerate(args) if i not in companions)
        drop = [tuple(args[:len(params)])] if len(args) > len(params) else []
        pack = [(list(args),), (tuple(args),)] if len(args) > 1 and len(required) == 1 else []
        # (collection, n, ...) calls usually pass lengths or bounds: drop them before packing.
        if args and isinstance(args[0], sized) and all(isinstance(a, int) for a in args[1:]):
            yield from drop + pack
        else:
            yield from pack + drop
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == len(required) > 1:
            yield tuple(args[0])
        if len(args) < len(required):
            first = next((a for a in args if hasattr(a, "__len__")), None)
            if first is not None and len(required) - len(args) == 1:
                yield tuple(args) + (len(first),)

    @_functools.wraps(_fn)
    def _wrapper(*args, **kwargs):
        # Recursive calls inside _fn resolve the global name: point it at _fn meanwhile.
        namespace = _fn.__globals__
        outer = namespace.get(_fn.__name__)
        namespace[_fn.__name__] = _fn
        try:
            if kwargs or _fits(args):
                return _fn(*args, **kwargs)
            # The call does not bind: try the plausible calling conventions in turn.
            for variant in _variants(args):
                if _fits(variant):
                    try:
                        trial = _copy.deepcopy(variant)  # a failed attempt must not mutate the caller's data
                    except Exception:
                        trial = variant
                    try:
                        return _fn(*trial)
                    except Exception:
                        continue
            return _fn(*args, **kwargs)
        finally:
            namespace[_fn.__name__] = outer

    return _wrapper


{entry} = _bind_arguments({entry})
"""

ALIASES = """
for _alias in {aliases!r}:
    globals().setdefault(_alias, {entry})
"""

HARNESS = r"""
import contextlib, io, json, math, signal, time

CANDIDATES = __CANDIDATES__
CALLS = __CALLS__
CHECKS = __CHECKS__
ENTRY = __ENTRY__


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def _bounded(fn, seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    return False


def _same(out, exp):
    try:
        if out == exp:
            return True
    except Exception:
        return False
    try:
        if _floats(exp):
            import numpy as np
            return bool(np.allclose(out, exp, rtol=1e-07, atol=1e-06))
    except Exception:
        pass
    return False


def _args(source):
    value = eval(source, {"inf": math.inf, "nan": math.nan})
    return value if isinstance(value, tuple) else (value,)


rows = []
SHARE = 20.0 / max(1, len(CANDIDATES))  # equal time for every candidate, whatever its order
for code in CANDIDATES:
    row = {"load": None, "calls": [], "checks": []}
    deadline = time.monotonic() + SHARE
    namespace = {"__name__": "candidate"}
    try:
        _bounded(lambda: exec(code, namespace), 3.0)
        fn = namespace[ENTRY]
    except BaseException as exc:
        row["load"] = type(exc).__name__
        rows.append(row)
        continue
    for source, expected in CHECKS:
        if time.monotonic() > deadline:
            row["checks"].append(None)
            continue
        try:
            out = _bounded(lambda: fn(*_args(source)), 1.0)
            row["checks"].append([_same(out, eval(expected)), repr(out)[:200]])
        except BaseException as exc:
            row["checks"].append([False, "raises " + type(exc).__name__])
    for source in CALLS:
        if time.monotonic() > deadline:
            row["calls"].append("skip")
            continue
        try:
            out = _bounded(lambda: fn(*_args(source)), 1.0)
            row["calls"].append("ok:" + repr(out)[:300])
        except BaseException as exc:
            row["calls"].append("raise:" + type(exc).__name__)
    rows.append(row)
print("@@CROSSCHECK@@" + json.dumps(rows))
"""

MARK = "@@CROSSCHECK@@"


def fenced_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S)]


def defines(code: str, entry: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry
        for n in tree.body
    )


def extract_code(text: str, entry: str, declared: str | None = None) -> str | None:
    """The first block that defines `entry`; failing that, one that defines the name the
    requirement itself declares (variants rename the target, e.g. `def candidate`), renamed."""
    blocks = fenced_blocks(text) + [text or ""]
    for block in blocks:
        if defines(block, entry):
            return block
    if declared and declared != entry:
        for block in blocks:
            if defines(block, declared):
                return renamed(block, declared, entry)
    return None


def renamed(code: str, old: str, new: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == old:
            node.name = new
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new
    return ast.unparse(tree)


def sanitize(code: str) -> str:
    """Keep imports, definitions and constant assignments; drop demos, prints, tests."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    keep = []
    for node in tree.body:
        if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(
            isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) in ("input", "print", "open")
            for sub in ast.walk(node)
        ):
            keep.append(node)
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            keep.append(node)  # module docstring
        elif isinstance(node, ast.Expr) and "setrecursionlimit" in ast.unparse(node):
            keep.append(node)
        elif isinstance(node, ast.Try) and all(
            isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body
        ):
            keep.append(node)
    try:
        return ast.unparse(ast.Module(body=keep, type_ignores=[]))
    except Exception:
        return code


def signature_of(code: str, entry: str) -> str:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        match = re.search(r"def\s+" + re.escape(entry) + r"\s*\(([^)]*)\)", code)
        return entry + "(" + (match.group(1).strip() if match else "...") + ")"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            return entry + "(" + ast.unparse(node.args) + ")"
    return entry + "(...)"


def name_variants(entry: str) -> list[str]:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", entry).lower()
    return sorted({entry.lower(), re.sub(r"_+", "_", snake)} - {entry})


def plain_question(text: str) -> str:
    text = " ".join((text or "").replace("`", "").split())
    text = re.sub(r"^(question|q)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def literal_source(value) -> str | None:
    try:
        source = repr(value)
        ast.literal_eval(source)
        return source
    except Exception:
        return None


def read_python_list(reply: str, name: str):
    for block in fenced_blocks(reply):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets
            ):
                try:
                    return ast.literal_eval(node.value)
                except Exception:
                    return None
    return None


def compact(text: str) -> str:
    return re.sub(r"[\s'\"]", "", str(text))


CONNECTOR = re.compile(
    r"\s*(?:==>|==|=>|->|\u279e|\u2192|returns?|should\s+return|should\s+be|gives|outputs?)\s*:?\s*",
    re.I,
)


def literal_calls(text: str, entry: str):
    """Yield (args, end) for every call `entry(<literal arguments>)` written in text."""
    for match in re.finditer(r"(?<![\w.])" + re.escape(entry) + r"\s*\(", text):
        start = match.end() - 1
        depth, quote, i = 0, None, start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        else:
            continue
        try:
            node = ast.parse("_f(" + text[start + 1 : i] + ")", mode="eval").body
            if node.keywords:
                continue
            yield tuple(ast.literal_eval(a) for a in node.args), i + 1
        except Exception:
            continue


def leading_literal(text: str):
    """Read a complete literal, never a literal prefix of an unsupported expression.

    A semicolon may start the next example and `, not ...` may contrast an output.
    Only these explicit prose boundaries can end a literal early. Operators and
    incomplete or oversized literals leave the example untrusted.
    """
    line = text.split("\n", 1)[0].strip()
    if not line or len(line) > 4096:
        return False, None

    def read(piece):
        try:
            return True, ast.literal_eval(piece)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return False, None

    ok, value = read(line)
    if ok:
        return ok, value
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(line).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif depth == 0:
                end = token.start[1]
                if token.string == ";" or (
                    token.string == "," and re.match(r",\s*not\b", line[end:], re.I)
                ):
                    return read(line[:end].strip())
    except (tokenize.TokenError, IndentationError):
        pass
    return False, None


def parse_examples(prompt: str, entry: str) -> list[tuple[str, str]]:
    """Examples written structurally: doctests and `entry(args) <connector> value`. Prose such as
    'for lst = [1, 2] the output should be 3' stays evidence in the prompt, never a check."""
    found = []
    for args, end in literal_calls(prompt, entry):
        rest = prompt[end:]
        connector = CONNECTOR.match(rest)
        if connector and connector.end() > 0 and rest[: connector.end()].strip():
            ok, value = leading_literal(rest[connector.end() :])
        else:
            line_start = prompt.rfind("\n", 0, end) + 1
            if (
                not prompt[line_start:end].lstrip().startswith(">>>")
                or rest.split("\n", 1)[0].strip()
            ):
                continue
            following = [line.strip() for line in rest.split("\n")[1:]]
            output = []
            for line in following:
                if not line or line.startswith(">>>") or line in ('"""', "'''", "```"):
                    break  # a docstring or fence closer ends the doctest output
                output.append(line)
            ok, value = leading_literal(" ".join(output)) if output else (False, None)
        pair = (literal_source(args), literal_source(value)) if ok else (None, None)
        if all(pair) and pair not in found:
            found.append(pair)
    return found[:8]


def declared_name(prompt: str, entry: str) -> str | None:
    """The function the requirement declares: the entry point itself, or the single other name
    it defines (variants rename the target, e.g. `def candidate(...)` for `unique`)."""
    if re.search(r"def\s+" + re.escape(entry) + r"\s*\(", prompt):
        return entry
    names = re.findall(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", prompt)
    return names[0] if len(set(names)) == 1 else None


def passed_count(results) -> int:
    return sum(1 for c in results or [] if c and c[0])


def edited(reply: str) -> str | None:
    if not reply or reply.strip().upper().startswith("UNCHANGED"):
        return None
    match = re.search(r"```(?:text)?[ \t]*\n(.*?)```", reply, re.S)
    return match.group(1).strip("\n") if match else None


def protected_requirement_lines(text: str) -> list[str]:
    """Declaration and example lines a deletion-only cleanup must preserve in order."""
    protected = []
    doctest_output = signature = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in ('"""', "'''", "```"):
            doctest_output = False
            continue
        example = bool(re.match(r"For\s+\w+\s*=", stripped))
        if stripped.startswith((">>>", "...")):
            example = doctest_output = True
        elif doctest_output:
            example = True
            doctest_output = False
        if re.match(r"(?:async\s+)?def\s+\w+\s*\(", stripped):
            signature = True
        if example or signature:
            protected.append(stripped)
        if signature and stripped.endswith(":"):
            signature = False
    return protected


def accept_deletion(original: str, new: str | None, names) -> str | None:
    """The cleaned text, or None unless it is a pure deletion of at most half the prose that keeps
    every doctest line, every function definition and every name the requirement defines."""
    if new is None or not new.strip() or new.strip() == original.strip():
        return None
    deleted = 0
    matcher = difflib.SequenceMatcher(a=original, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            deleted += i2 - i1
        elif tag in ("replace", "insert") and (original[i1:i2].strip() or new[j1:j2].strip()):
            return None
    if deleted > 0.5 * len(original) or original.count(">>>") != new.count(">>>"):
        return None
    remaining = iter(line.strip() for line in new.splitlines())
    for protected in protected_requirement_lines(original):
        if not any(line == protected for line in remaining):
            return None
    for name in names:
        if name and any(
            pair not in parse_examples(new, name) for pair in parse_examples(original, name)
        ):
            return None
    for defined in re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", original):
        if not re.search(r"def\s+" + re.escape(defined) + r"\s*\(", new):
            return None
    for name in names:
        if not name:  # the requirement may declare no function
            continue
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, original) and not re.search(pattern, new):
            return None
    return new


class CrossCheckClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "drafts": 3,  # first implementations, written before asking
        "finals": 3,  # implementations written with the user's answer
        "probes": 10,  # LLM-written probe inputs for agreement
        "repairs": 2,  # counterexample-guided repair rounds on the selected candidate
        "question_style": "confirm_alt",  # "confirm_alt" (offers an alternative), "confirm", "neutral" or "open"
        "adapter": True,  # bind-checked argument packing/unpacking/arity at call time
        "aliases": True,  # lower/snake-case spellings of the entry point
        "llm_timeout": 100,  # seconds to wait for one batch of parallel LLM calls
        # "always" asks; "never" skips (measurement only); "gate" skips only when every draft
        # satisfies every stated example (at least gate_min_examples of them) and all drafts
        # agree on every probe input: a question there mostly risks an invented answer.
        "ask_policy": "always",
        "gate_min_examples": 1,
        # Up to this many alternative readings, one extra draft each (0 disables); with
        # reading_question "inform" the question writer sees which readings drafts follow.
        "readings": 0,
        "reading_question": "inform",
        # Re-read the chosen implementation against the user's answer and fix contradictions;
        # a fix is kept only if it passes at least as many trusted checks.
        "verify": True,
        # Drop a check that no final satisfies: every final was written after the answer, so
        # an example all of them contradict is one the answer overrides (or a corrupted one),
        # and repairing toward it would undo the answer.
        "check_filter": True,
        # Delete a sentence that cannot hold together with the rest before drafting; the edit is
        # kept only if it is a pure deletion (examples, signature and names untouched).
        "preclean": True,
        # v1.16 candidate: one more question after the first answer when the run allows a second
        # turn. "always" asks it; "off", or a one-turn limit, reproduces v1.14rc3 exactly.
        # "gated" (v1.20) asks the second question only when a majority of three classifications
        # says the first answer CORRECTED the assumption; "always" (= v1.16) asks it whenever a second
        # turn exists and only logs that vote.
        "second_question": "gated",
        "second_question_start_limit": 100,  # seconds since the task started
        "task_deadline": 250,  # seconds; bounds the second round below the 300 s SDK watchdog
    }

    # LLM / sandbox helpers --------------------------------------------------------------

    def ask_llm(self, env, prompt: str) -> str:
        return self.parallel(env, [prompt])[0]

    def parallel(self, env, prompts: list[str], timeout: float | None = None) -> list[str]:
        """Run LLM calls concurrently; a call that has not returned by the deadline counts as
        empty, so one hung request cannot stall the task past the runner's watchdog."""
        results = [""] * len(prompts)

        def call(index, prompt):
            if env.prompt_cost >= 0.9 * env.prompt_budget:
                return
            try:
                results[index] = env.llm([{"role": "user", "content": prompt}]) or ""
            except Exception:  # budget exhausted or provider failure: degrade, never crash
                results[index] = ""

        threads = [
            threading.Thread(target=call, args=(i, p), daemon=True) for i, p in enumerate(prompts)
        ]
        deadline = time.monotonic() + float(
            self.config["llm_timeout"] if timeout is None else timeout
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return list(results)

    def finish(self, code: str, entry: str) -> str:
        code = sanitize(code)
        if self.config["adapter"]:
            code = code.rstrip() + "\n" + ADAPTER.replace("{entry}", entry)
        variants = name_variants(entry) if self.config["aliases"] else []
        if variants:
            code = code.rstrip() + "\n" + ALIASES.format(aliases=variants, entry=entry)
        return code

    def run_harness(self, env, codes: list[str], entry: str, calls: list[str], checks: list[tuple]):
        if not codes:
            return None
        script = (
            HARNESS.replace("__CANDIDATES__", repr(codes))
            .replace("__CALLS__", repr(calls))
            .replace("__CHECKS__", repr([list(c) for c in checks]))
            .replace("__ENTRY__", repr(entry))
        )
        try:
            output = env.exec_code(script)
        except Exception:
            return None
        marker = output.rfind(MARK)
        if marker < 0:
            return None
        try:
            rows = json.loads(output[marker + len(MARK) :].strip().splitlines()[0])
        except (ValueError, IndexError):
            return None
        return rows if len(rows) == len(codes) else None

    # Evidence -----------------------------------------------------------------------------

    def probe_prompt(self, prompt: str, entry: str, signature: str, context: str = "") -> str:
        return PROBE_PROMPT.format(
            count=int(self.config["probes"]),
            entry=entry,
            prompt=prompt,
            signature=signature,
            context=context,
        )

    def read_calls(self, reply: str) -> list[str]:
        inputs = read_python_list(reply, "INPUTS") or []
        calls = []
        for args in inputs if isinstance(inputs, list) else []:
            args = args if isinstance(args, tuple) else (args,)
            source = literal_source(args)
            if source and source not in calls:
                calls.append(source)
        return calls[: int(self.config["probes"]) + 2]

    @staticmethod
    def describe(rows, drafts, entry, calls, examples) -> str:
        lines = []
        if not examples:
            lines.append("The requirement states no concrete examples.")
        elif rows:
            failing = []
            for i, (args, expected) in enumerate(examples):
                results = [
                    r["checks"][i]
                    for r in rows
                    if not r.get("load") and i < len(r["checks"]) and r["checks"][i] is not None
                ]
                # One draft failing where another succeeds is a bug, not missing information.
                if results and not any(ok for ok, _ in results):
                    failing.append(
                        f"- no draft satisfies the stated example {entry}{args} -> expected "
                        f"{expected}; the drafts return {results[0][1]}"
                    )
            lines += failing[:3] or [
                f"Some draft satisfies each of the {len(examples)} stated example(s)."
            ]
        disputed = []
        if rows:
            for c, source in enumerate(calls):
                outs = {}
                for row in rows:
                    if not row.get("load") and c < len(row["calls"]) and row["calls"][c] != "skip":
                        outs.setdefault(row["calls"][c], 0)
                        outs[row["calls"][c]] += 1
                if len(outs) > 1:
                    shown = "; ".join(
                        f"{k.split(':', 1)[1] if k.startswith('ok:') else k} ({v} draft{'s' if v > 1 else ''})"
                        for k, v in sorted(outs.items(), key=lambda kv: -kv[1])
                    )
                    disputed.append(f"- {entry}{source}: {shown}")
        lines += (
            (["Drafts disagree on:"] + disputed[:3])
            if disputed
            else ["All drafts agree on the probe inputs."]
        )
        signatures = sorted({signature_of(d, entry) for d in drafts})
        if len(signatures) > 1:
            lines.append("Drafts use different signatures: " + " / ".join(signatures))
        return "\n".join(lines)

    @staticmethod
    def describe_readings(rows, readings, entry, calls) -> str:
        """Which readings produce behaviour different from the first draft, and where."""
        lines = ["Alternative readings of the requirement (drafts were written for each):"]
        base = rows[0] if rows and not rows[0].get("load") else None
        for reading, index in readings:
            row = rows[index] if index < len(rows) else None
            where = ""
            if base and row and not row.get("load"):
                for c, source in enumerate(calls):
                    if (
                        c < len(base["calls"])
                        and c < len(row["calls"])
                        and base["calls"][c] != row["calls"][c]
                    ):
                        where = f" (differs from your draft at {entry}{source}: {row['calls'][c]} vs {base['calls'][c]})"
                        break
                where = where or " (same behaviour as your draft on the probes)"
            lines.append(f"- {reading}{where}")
        return "\n".join(lines)

    # Main --------------------------------------------------------------------------------

    def run(self, env, problem):
        started = time.monotonic()
        prompt, entry = problem["prompt"], problem["entry_point"]
        if self.config["preclean"]:
            # The user still sees the original text; only our own steps read the cleaned one.
            try:
                reply = self.ask_llm(env, PRECLEAN_PROMPT.format(entry=entry, prompt=prompt))
                names = [entry, declared_name(prompt, entry)]
                prompt = accept_deletion(prompt, edited(reply), names) or prompt
            except Exception:  # cleaning is optional: degrade to the original text, never crash
                pass
        declared = declared_name(prompt, entry)
        has_signature = declared is not None
        rules = RULES.format(entry=entry, signature_rule="" if has_signature else SIGNATURE_RULE)
        fallback = f"def {entry}(*args, **kwargs):\n    return None\n"

        # 1. Drafts, stated examples and probe inputs, in parallel.
        k = max(1, int(self.config["drafts"]))
        given = (
            (entry + signature_of(prompt, declared)[len(declared) :])
            if has_signature
            else "not given; infer it from the description"
        )
        replies = self.parallel(
            env,
            [DRAFT_PROMPT.format(prompt=prompt, rules=rules)] * k
            + [self.probe_prompt(prompt, entry, given)]
            + (
                [
                    READINGS_PROMPT.format(
                        count=int(self.config["readings"]), entry=entry, prompt=prompt
                    )
                ]
                if int(self.config["readings"]) > 0
                else []
            ),
        )
        examples = parse_examples(prompt, declared or entry)
        calls = self.read_calls(replies[k])
        drafts = [sanitize(c) for c in (extract_code(r, entry, declared) for r in replies[:k]) if c]
        readings = []
        if int(self.config["readings"]) > 0:
            listed = read_python_list(replies[k + 1], "READINGS") or []
            readings = [str(r) for r in listed if isinstance(r, str) and r.strip()][
                : int(self.config["readings"])
            ]
            extra = self.parallel(
                env, [READING_DRAFT.format(reading=r, prompt=prompt, rules=rules) for r in readings]
            )
            placed = []
            for reading, reply in zip(readings, extra, strict=False):
                code = extract_code(reply, entry, declared)
                if code:
                    placed.append((reading, len(drafts)))
                    drafts.append(sanitize(code))
            readings = placed
        best = drafts[0] if drafts else fallback

        # 2. Run the drafts on the stated examples and on the probe inputs.
        rows = self.run_harness(
            env, [self.finish(d, entry) for d in drafts], entry, calls, examples
        )
        evidence = self.describe(rows, drafts, entry, calls, examples)
        if readings and rows:
            evidence += "\n" + self.describe_readings(rows, readings, entry, calls)
        explained = {
            i
            for i in range(len(examples))
            for r in rows or []
            if not r.get("load") and i < len(r["checks"]) and r["checks"][i] and r["checks"][i][0]
        }
        if rows:
            # Ask about the draft that explains the most stated examples.
            scores = [
                sum(1 for c in r["checks"] if c and c[0]) if not r.get("load") else -1 for r in rows
            ]
            best = drafts[max(range(len(drafts)), key=lambda i: (scores[i], -i))]

        # 3. One question, unless the evidence leaves nothing to ask about.
        question, answer = None, None
        if self.should_ask(rows, examples, calls) and env.can_ask():
            question = plain_question(
                self.ask_llm(
                    env,
                    QUESTION_PROMPT.format(
                        entry=entry,
                        prompt=prompt,
                        draft=best,
                        evidence=evidence,
                        phrasing=PHRASINGS[self.config["question_style"]],
                    ),
                )
            )
            if self.config["question_style"] == "confirm":
                # Never offer an alternative the user could pick without information.
                question = re.sub(r"(Is that (?:right|correct))\b.*$", r"\1?", question, flags=re.I)
            if question and not re.search(r"\b" + re.escape(entry) + r"\b", question):
                # The user only sees the requirement text, which may not name the function.
                question = f"About {entry}: {question}"
            if question:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    answer = None
                except Exception:
                    answer = None

        if not answer:
            # Without an answer the drafts are the final candidates.
            code, _ = self.pick(drafts, rows) if drafts else (best, None)
            return "```python\n" + self.finish(code, entry) + "\n```"

        # 4-5. Final implementations, trusted checks, selection, repair and verify.
        final_prompt = FINAL_PROMPT.format(
            prompt=prompt, question=question, answer=answer, rules=rules
        )
        context = f"\nThe user clarified:\nQ: {question}\nA: {answer}\n"
        code, finals, calls = self.post_answer(
            env,
            prompt,
            entry,
            declared,
            rules,
            examples,
            explained,
            calls,
            best,
            final_prompt,
            question,
            answer,
            context,
        )
        delivered = self.finish(code, entry)
        note = getattr(env, "note", None)  # local logging only; the SDK environment has none
        if note:
            note("k1_delivered", delivered)
        second = self.second_round(
            env,
            started,
            note,
            prompt,
            entry,
            declared,
            rules,
            examples,
            explained,
            calls,
            finals,
            code,
            question,
            answer,
        )
        return "```python\n" + (second or delivered) + "\n```"

    def post_answer(
        self,
        env,
        prompt,
        entry,
        declared,
        rules,
        examples,
        explained,
        calls,
        best,
        final_prompt,
        question,
        answer,
        context,
        deadline=None,
    ):
        """Finals, trusted checks and probes in parallel; select; repair; verify. Returns the
        chosen code (unfinished), the finals and the probe calls. Without a deadline this is
        exactly the v1.14rc3 post-answer stage."""
        n = max(1, int(self.config["finals"]))
        numbered = (
            "\n".join(f"{i + 1}. {entry}{a} -> {e}" for i, (a, e) in enumerate(examples))
            or "(none)"
        )
        timeout = None
        if deadline is not None:
            timeout = max(5.0, min(float(self.config["llm_timeout"]), deadline - time.monotonic()))
        outputs = self.parallel(
            env,
            [final_prompt] * n
            + [
                CHECKS_PROMPT.format(
                    entry=entry, prompt=prompt, question=question, answer=answer, examples=numbered
                ),
                self.probe_prompt(prompt, entry, signature_of(best, entry), context),
            ],
            timeout,
        )
        finals = [sanitize(c) for c in (extract_code(r, entry, declared) for r in outputs[:n]) if c]
        checks = self.trusted_checks(outputs[n], examples, explained)
        calls = self.read_calls(outputs[n + 1]) or calls
        if not finals:
            finals = [best]
        code, results, checks = self.select(env, finals, entry, calls, checks)

        # Counterexample-guided repair against the trusted checks.
        for _ in range(int(self.config["repairs"]) if checks else 0):
            if results is None:
                break
            if deadline is not None and time.monotonic() > deadline - 100:
                break
            failures = [
                f"- {entry}{a} should return {e}, got {check[1]}"
                for (a, e), check in zip(checks, results, strict=False)
                if check and not check[0]
            ]
            if not failures:
                break
            repaired = extract_code(
                self.ask_llm(
                    env,
                    REPAIR_PROMPT.format(
                        prompt=prompt,
                        clarification=context,
                        code=code,
                        failures="\n".join(failures[:4]),
                        rules=rules,
                    ),
                ),
                entry,
                declared,
            )
            if not repaired:
                break
            repaired = sanitize(repaired)
            rows = self.run_harness(env, [self.finish(repaired, entry)], entry, [], checks)
            if not rows or rows[0].get("load"):
                break
            if passed_count(rows[0]["checks"]) <= passed_count(results):
                break
            code, results = repaired, rows[0]["checks"]

        if self.config["verify"] and (deadline is None or time.monotonic() < deadline - 100):
            code = self.verify(env, prompt, entry, question, answer, code, checks, results)
        return code, finals, calls

    def classify(self, env, question, answer):
        """Majority of three samples: CONFIRM, CORRECT or ADD; None without a majority."""
        replies = self.parallel(
            env, [NEED_PROMPT.format(question=question, answer=answer)] * 3, 30.0
        )
        labels = [
            next(
                (
                    w
                    for w in ("CONFIRM", "CORRECT", "ADD")
                    if (r or "").strip().upper().startswith(w)
                ),
                None,
            )
            for r in replies
        ]
        counts = {w: labels.count(w) for w in ("CONFIRM", "CORRECT", "ADD")}
        best = max(counts, key=counts.get)
        return (best if counts[best] >= 2 else None), labels

    def second_round(
        self,
        env,
        started,
        note,
        prompt,
        entry,
        declared,
        rules,
        examples,
        explained,
        calls,
        finals,
        code,
        question,
        answer,
    ):
        """One more question about a different point, then the post-answer stage again with both
        answers. Returns the finished code, or None to keep the one-question result."""
        mode = self.config["second_question"]
        if mode not in ("always", "gated") or not env.can_ask():
            return None
        if time.monotonic() - started > float(self.config["second_question_start_limit"]):
            if note:
                note("skip", "time")
            return None
        verdict, votes = self.classify(env, question, answer)
        if note:
            note("need", votes)
        if mode == "gated" and verdict != "CORRECT":
            return None
        deadline = started + float(self.config["task_deadline"])
        rows = self.run_harness(
            env, [self.finish(f, entry) for f in finals], entry, calls, examples
        )
        evidence = self.describe(rows, finals, entry, calls, examples)
        for old, new in (
            ("Drafts", "Implementations"),
            ("drafts", "implementations"),
            ("draft", "implementation"),
        ):
            evidence = evidence.replace(old, new)
        reply, gate = self.parallel(
            env,
            [
                QUESTION2_PROMPT.format(
                    entry=entry,
                    prompt=prompt,
                    question=question,
                    answer=answer,
                    code=code,
                    evidence=evidence,
                ),
                GATE_PROMPT.format(entry=entry, prompt=prompt, question=question, answer=answer),
            ],
            60.0,
        )
        if note:
            note("gate", gate)
        question2 = plain_question(reply)
        if question2 and not re.search(r"\b" + re.escape(entry) + r"\b", question2):
            question2 = f"About {entry}: {question2}"
        if not question2:
            if note:
                note("skip", "no question")
            return None
        try:
            answer2 = env.ask_human(question2)
        except Exception:
            answer2 = None
        if not answer2:
            if note:
                note("skip", "no answer")
            return None
        both = f"{answer}\nQ: {question2}\nA: {answer2}"
        code2, _, _ = self.post_answer(
            env,
            prompt,
            entry,
            declared,
            rules,
            examples,
            explained,
            calls,
            code,
            FINAL2_PROMPT.format(
                prompt=prompt,
                question=question,
                answer=answer,
                question2=question2,
                answer2=answer2,
                rules=rules,
            ),
            question,
            both,
            f"\nThe user clarified:\nQ: {question}\nA: {both}\n",
            deadline,
        )
        return self.finish(code2, entry)

    def verify(self, env, prompt, entry, question, answer, code, checks, results):
        reply = self.ask_llm(
            env,
            VERIFY_PROMPT.format(
                prompt=prompt, question=question, answer=answer, entry=entry, code=code
            ),
        )
        if not reply or reply.strip().upper().startswith("OK"):
            return code
        fixed = extract_code(reply, entry, declared_name(prompt, entry))
        if not fixed:
            return code
        fixed = sanitize(fixed)
        if checks:
            rows = self.run_harness(env, [self.finish(fixed, entry)], entry, [], checks)
            if (
                not rows
                or rows[0].get("load")
                or passed_count(rows[0]["checks"]) < passed_count(results)
            ):
                return code
        return fixed

    def should_ask(self, rows, examples, calls) -> bool:
        policy = self.config["ask_policy"]
        if policy != "gate":
            return policy != "never"
        live = [r for r in rows or [] if not r.get("load")]
        if len(live) < 2 or len(examples) < int(self.config["gate_min_examples"]):
            return True
        if any(passed_count(r["checks"]) < len(examples) for r in live):
            return True
        outputs = [tuple(o if o.startswith("ok:") else "fail" for o in r["calls"]) for r in live]
        return "fail" in outputs[0] or any(o != outputs[0] for o in outputs[1:]) or not calls

    def trusted_checks(self, reply: str, examples: list[tuple[str, str]], explained: set[int]):
        """Stated examples the answer confirms or leaves consistent with the text; never an example
        the text contradicts (variants often corrupt examples); plus calls the answer settles."""
        verdicts = read_python_list(reply, "VERDICTS")
        confirmed = read_python_list(reply, "CONFIRMED") or []
        checks = []
        for i, example in enumerate(examples):
            verdict = (
                str(verdicts[i]).upper()
                if isinstance(verdicts, list) and i < len(verdicts)
                else "KEEP"
            )
            if verdict.startswith("DROP") or verdict.startswith("CONFLICT"):
                continue  # an example the text contradicts is evidence, not a target
            checks.append(example)
        for item in confirmed if isinstance(confirmed, list) else []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                args = item[0] if isinstance(item[0], tuple) else (item[0],)
                pair = (literal_source(args), literal_source(item[1]))
                if all(pair) and pair not in checks:
                    checks.append(pair)
        return checks[:10]

    def select(self, env, finals: list[str], entry: str, calls: list[str], checks: list[tuple]):
        """Run the candidates on probes and checks, then pick one (see pick). Returns the code,
        its check results and the checks still trusted (see check_filter)."""
        if len(finals) == 1 and not checks:
            return finals[0], None, checks
        rows = self.run_harness(env, [self.finish(c, entry) for c in finals], entry, calls, checks)
        if rows and self.config["check_filter"]:
            live = [r for r in rows if not r.get("load")]
            keep = [
                j
                for j in range(len(checks))
                if any(j < len(r["checks"]) and r["checks"][j] and r["checks"][j][0] for r in live)
            ]
            checks = [checks[j] for j in keep]
            rows = [
                r if r.get("load") else {**r, "checks": [r["checks"][j] for j in keep]}
                for r in rows
            ]
        code, results = self.pick(finals, rows)
        return code, results, checks

    @staticmethod
    def pick(candidates: list[str], rows):
        """Most checks passed, then most execution agreement, then first written. Every candidate
        gets the same time share, so a check or probe it did not finish counts against it.
        Returns the chosen code and its check results (None when nothing ran)."""
        if not rows:
            return candidates[0], None
        pool = [i for i, r in enumerate(rows) if not r.get("load")]
        if not pool:
            return candidates[0], None
        passed = {i: passed_count(rows[i]["checks"]) for i in pool}
        top = max(passed.values())
        pool = [i for i in pool if passed[i] == top]
        signatures = {
            i: tuple(o if o.startswith("ok:") else (i, o) for o in rows[i]["calls"]) for i in pool
        }
        support = {i: sum(signatures[i] == signatures[j] for j in pool) for i in pool}
        best = max(pool, key=lambda i: (support[i], -i))
        return candidates[best], rows[best]["checks"]
