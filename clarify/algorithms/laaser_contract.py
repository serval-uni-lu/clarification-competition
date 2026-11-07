# SPDX-FileCopyrightText: 2026 Matias Brizzio <matias.brizzio@list.lu>
#
# SPDX-License-Identifier: MIT

"""LAASeR_Contract: one contract question, written from a Pinpoint audit, z3 and a draft.

The task text may be vague, incomplete or contradictory; hidden tests compare the function's
return values with what the user really intends. We may ask ONE question. The pipeline:

1. Pinpoint audit (`pinpoint_audit`). One LLM call lists every defect of the task
   (CONTRADICTION, VAGUE, MISSING, ODD, each with two competing readings and its impact) and
   decides ASK or CODE. A shape reference for the data structures the task mentions (pairs or
   N-tuples, flat or nested results, ...) is shown as an optional aid. When the task states a
   return format, the audit is told to contrast it with the type the function would naturally
   return, and the coding rules say the user's answer overrides the text, the signature and
   the examples.

2. z3 evidence (`z3_evidence`). The examples the task states are parsed and their argument
   and result types inferred; a type hint an example violates is a conflict found without a
   solver. With z3 installed, one LLM call formalizes the task's statements as SMT-LIB
   constraints over the function's inputs and output, and z3 checks them against the stated
   examples: an unsatisfiable core that contains a statement and an example proves they
   cannot both hold (and, when the statements force a value, which value). A proven conflict is
   evidence for the question, not a guess. When no example conflicts, one more call formalizes
   the two readings of each audit defect as SMT-LIB functions, and z3 searches a concrete input
   on which they return different values. Without z3 this step finds only type clashes.

3. Contract question (`contract_question`). One LLM call writes a draft without asking
   anything; a second writes the question from the draft, the audit's defects and the z3
   evidence. The question states the draft's contract on ordinary inputs (the arguments with
   their types and order, the exact return type and format, the rule, one small example call
   with its result) and asks whether it is right or should instead do the most plausible
   alternative; a proven conflict is settled first, and a stated example that contradicts the
   text is asked about alone. The user knows every hidden requirement, so one answer corrects
   every part of the contract that is wrong, not only the one point a two-reading question
   settles. On tasks that show no signature the question names the argument order, which tests
   call positionally. With `contract_drafts` > 1, several drafts run in the sandbox first on
   the stated examples and on probe inputs; the question is about the one that satisfies most
   examples and sees the calls on which the drafts disagree. If no contract question can be
   written (no usable draft), the question falls back to the audit's: two more audits are
   sampled and a judge picks among their questions by the README's criteria; on tasks that
   state a return format, a question contrasting that format with the natural type wins.

4. Code with the answer (`code_with_answer`). The answer is authoritative; a refusal or a
   dodge is not used. The code is checked (a python block, the entry point, no placeholder or
   constant body) and repaired up to twice.

5. Verification by execution (`verify_by_execution`). Several implementations run in the
   sandbox on probe calls written from the task and the answer: crashes that look like bugs
   are repaired, results the answer rules out are arbitrated by an independent step-by-step
   check, and the implementation with the fewest problems, then the behaviour most of them
   share, is kept.

6. Signature fixes (`fix_signature`). On tasks that show no signature: a (scalar, collection)
   parameter pair is swapped to (collection, scalar), the order tests use, unless the answer
   talks about the order; and the entry point is wrapped so that a call that does not bind to
   its signature (a length passed after its collection, extra trailing arguments, arguments to
   pack or unpack, a missing length) is retried with a generic alternative. A call that binds
   runs unchanged.

Interaction with the benchmark is only through `env.llm`, `env.ask_human` (once) and
`env.exec_code`. Two options control z3 (`z3-solver`):

- `use_solver` (default True): False, or z3 not installed, skips the solver everywhere (only
  type clashes, which need no solver, remain) and keeps everything else.
- `z3_mode` (default "evidence"): z3 runs on every task and its proven conflict feeds the
  contract question, which is always asked. "on_code": the contract question is asked only
  when the audit asks; when the audit would code, z3 looks for a proven conflict and asks
  about it, else the task is coded without a question.

Each task has a wall-clock budget (`time_budget`, 240 s; at most 90 s per LLM call): the runner
restarts a whole batch when one task passes 300 s, so z3 and the verification are skipped when
little time is left.

Team: LAASeR - LIST
Team Members: Matias Brizzio, Jordi Cabot, Renzo Degiovanni
Main Contact: matias.brizzio@list.lu
"""

from __future__ import annotations

import ast
import base64
import math
import operator
import random
import re
import threading
import time
from collections import Counter
from fractions import Fraction
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, LimitsExceededException, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

try:
    import z3
except ImportError:  # optional: without z3 the solver step is skipped
    z3 = None


# ---------------------------------------------------------------------------------------------
# Prompt helpers and limits
# ---------------------------------------------------------------------------------------------
# _fill substitutes {placeholders} in one pass (task text and model output contain braces);
# CODING_RULES are appended to every coding prompt.

MAX_REPAIR_ATTEMPTS = 2


MIN_QUESTION_LENGTH = 10


MAX_QUESTION_LENGTH = 600


MIN_CUT_ANSWER = 30  # a reply cut at a backtick shorter than this carries no fact


CODING_RULES = """
Rules for the implementation:
1. Name the function exactly `{entry_point}`, even if the task text uses another name. Keep the
   parameters shown in the task; if none are shown, use the most natural parameters for the
   description.
2. The hidden tests only compare return values. Never weaken or complicate the logic to satisfy
   requirements they cannot observe (time complexity, "no loops", "no recursion", style).
3. Where the task is still ambiguous or contradictory, choose the most standard, least surprising
   behaviour for a function with this name and purpose.
4. Write only the function (plus any imports and helpers it needs) in a single ```python block:
   no prints, no input(), no example usage.
""".strip()


PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


def _fill(template: str, **fields: str) -> str:
    # One pass over the template's own placeholders: the inserted values (task text, model or
    # user output) routinely contain literal braces and must never be substituted into.
    return PLACEHOLDER_PATTERN.sub(lambda m: fields.get(m.group(1), m.group(0)), template)


def _rules(entry_point: str) -> str:
    return _fill(CODING_RULES, entry_point=entry_point)


# ---------------------------------------------------------------------------------------------
# 1. Pinpoint audit: defects of the task and the ASK / CODE decision
# ---------------------------------------------------------------------------------------------
# AUDIT_TEMPLATE lists the task's defects with competing readings and decides ASK or CODE.
# The shape reference (QUESTION_BANK) adds typical shape questions for the data structures the
# task mentions; FORMAT_CONTRAST_RULE applies when the task states a return format. When no
# contract question can be written, more audits are sampled and judge_questions picks one by
# the README's criteria (criticality, search space, leakage, atomicity, objectivity).

AUDIT_TEMPLATE = """
You are about to implement a Python function for a user. The task text below may have been
edited so that it is vague, incomplete or self-contradictory. Hidden tests will call
`{entry_point}` and compare its return values with the behaviour the user really intends.
Before coding you may ask the user ONE clarifying question. A question is cheap, but
implementing the wrong behaviour fails the tests.

Task:
<<<
{prompt}
>>>

STEP 1 - DEFECTS. List every place where the task does not uniquely determine the values that
`{entry_point}` must return. Check in particular for:
- CONTRADICTION: two statements, or a statement and an example / signature / type hint, that
  cannot both hold.
- VAGUE: a word or phrase whose interpretation changes the output (e.g. "similar", "certain",
  "okay", "large", "standard").
- MISSING: something the code needs that the text never fixes: the arguments and their
  structure, the return type or format, a formula, a threshold, a tie-break rule, the result
  for a special input.
- ODD: an unusual extra requirement that looks out of place for this function.
Write one line per defect:
- <TYPE> | "<exact phrase from the task, or: not stated>" | reading A: ... | reading B: ... | IMPACT: HIGH or LOW
IMPACT is HIGH when the two readings return different values on ordinary inputs, and LOW when
they differ only on rare edge inputs. Requirements that tests cannot observe (time complexity,
"no loops", "no recursion", style, comments) are never defects. If there are none, write: - none

STEP 2 - DECISION.
If no defect is HIGH and the task (with its examples) fixes both the arguments and the exact
return format, write the line `DECISION: CODE` followed by the complete implementation.
Otherwise write the line `DECISION: ASK` followed by exactly one line:
QUESTION: <your question>
Ask about the HIGH defect that would make the most tests fail (contradictions and the return
format first, then missing rules and vague terms, edge cases last). Quote or name the phrase
and offer the competing readings, so that the complete answer is exactly one fact. Example:
QUESTION: The task says "a frequency similar to the value" - should the frequency be exactly equal to the value, or at least the value?
Ask about one thing only, and never ask for test cases or for the implementation.

{rules}
""".strip()


QUESTION_BANK = (
    (
        r"\btuples?\b|\bpairs?\b",
        "Tuples: a pair or any length? Is the result one flat tuple, a tuple of tuples or a list"
        " of tuples, e.g. (1, 2, 3, 4) or ((1, 2), (3, 4))?",
    ),
    (
        r"\blists?\b|\barrays?\b|\bsequences?\b|\bcollections?\b",
        "Lists: flat or nested (a list of lists, of tuples or of pairs)? Is the result a list, a"
        " tuple or a set?",
    ),
    (
        r"\bmatri(?:x|ces)\b|\bgrids?\b|\btables?\b|\brows?\b|\bcolumns?\b",
        "Matrices: a list of rows? Rows of equal length? Are positions given as (row, column)?",
    ),
    (
        r"\btrees?\b|\broots?\b|\bleaf\b|\bleaves\b|\bchildren\b",
        "Trees: binary or N-ary? Given as nested lists or tuples, a dict of children, or node"
        " objects?",
    ),
    (
        r"\bgraphs?\b|\bedges?\b|\bvert(?:ex|ices)\b|\badjacen|\bneighbou?rs?\b",
        "Graphs: adjacency lists, a matrix, a dict or a list of edge pairs? Directed or not?",
    ),
    (
        r"\bdict(?:ionary|ionaries)?\b|\bmappings?\b|\bkey-value\b",
        "Mappings: a dict or a list of (key, value) pairs? Of which key and value types?",
    ),
    (
        r"\bsets?\b(?! bits?\b)",
        "Sets: a set, or a list without duplicates? In which order?",
    ),
    (
        r"\bstrings?\b|\bwords?\b|\bsentences?\b|\bcharacters?\b",
        "Text as a sequence: a string, or a list of words or characters, as input or as result?",
    ),
)


NUMBER_ENTRY = (
    r"\bdigits?\b",
    "Digits of negative numbers: is the sign ignored, or does it count with the first digit,"
    " e.g. -12 -> 1 + 2 or -1 + 2?",
)


RETURN_FORMAT_STATEMENT = re.compile(
    r"\b(?:return|output|deliver|give)\w*\b[^.\n]{0,60}\b(?:as a string|as strings?|string"
    r" representation|as an integer|as a tuple|as a list|as a boolean|binary number|should be a"
    r" string|must be an integer|needs to be an integer)",
    re.IGNORECASE,
)


FORMAT_TOPIC = re.compile(
    r"\b(?:as a string|a string|the string|string representation|return type|returned as"
    r"|return (?:a|an) (?:list|integer|int|string|tuple|bool|boolean|float)|integer or|list or"
    r"|or a list|or an integer|or a string|tuple|boolean|data type"
    r"|type of the (?:output|result|return))\b",
    re.IGNORECASE,
)


FORMAT_CONTRAST_RULE = """RETURN FORMAT. The task states a format for the return value, and that statement may be wrong.
The hidden tests compare return values, so if you ask, ask about the type of the return value:
one reading is the format the task states, the other the type the function would naturally
return without that statement, both as Python literals for the same call, e.g. QUESTION: Should
total(5) return the integer 15 or the string '15'? Requirements the tests cannot observe (loops,
time complexity) are never worth the question.
"""


TYPED_READING = re.compile(
    r"\b(?:integers?|int|floats?|lists?|tuples?|booleans?|bool|True|False|dict"
    r"|dictionary)\b",
    re.IGNORECASE,
)


STRING_READING = re.compile(r"\bstrings?\b|'[^']*'|\"[^\"]*\"", re.IGNORECASE)


ANSWER_OVERRIDES_RULE = """5. If the user answered a question, the answer overrides the task text, the signature, the type
   hints and the examples. If the answer states the return type, return exactly that type and
   change the signature to match."""


FORMAT_QUESTION_TEMPLATE = """The Python task below states a format for the return value of `{entry_point}`, and that
statement may be wrong: the hidden tests compare return values with what the user really
intends. Write ONE question for the user that settles the type of the return value. One reading
is the format the task states; the other is the type `{entry_point}` would naturally return
without that statement. Show both as Python literals for the same call, e.g.
QUESTION: Should total(5) return the integer 15 or the string '15'?
No backticks, one question only.

Task:
<<<
{prompt}
>>>

Answer with exactly one line:
QUESTION: <your question>"""


MAX_CHECKLIST = 4


CHECKLIST_HEADER = """SHAPE REFERENCE (optional). Use it only if a data structure in the task has a shape that the
text and the examples do not make clear to you; otherwise ignore it. Typical shape questions:"""


SHAPE_QUESTION_RULE = """When the question is about the shape or format of a value, show the competing readings as
Python literals for one small input, e.g. QUESTION: For [1, 2, 3, 4], should the result be
[(1, 2), (3, 4)] or [(1, 2), (2, 3), (3, 4)]?
"""


AUDIT_CHECKLIST_TEMPLATE = AUDIT_TEMPLATE.replace(
    "Write one line per defect:\n", "{checklist}Write one line per defect:\n", 1
).replace(
    "Ask about one thing only, and never ask for test cases or for the implementation.\n",
    SHAPE_QUESTION_RULE
    + "Ask about one thing only, and never ask for test cases or for the implementation.\n",
    1,
)


def contrastive(question: str) -> bool:
    """The question sets a typed reading (integer, list, True ...) against a string reading."""
    return bool(TYPED_READING.search(question) and STRING_READING.search(question))


def format_type_question(
    env: ClarificationEnvironment, prompt: str, entry_point: str
) -> str | None:
    """A contrastive question about the return type, written by one call (None if it is not)."""
    try:
        reply = env.llm(_fill(FORMAT_QUESTION_TEMPLATE, prompt=prompt, entry_point=entry_point))
    except Exception:
        return None
    question = atomic_question(_parse_question(reply))
    return question if question and contrastive(question) else None


def checklist_block(prompt: str, numbers: bool = False) -> str:
    """The shape reference for the data structures the task mentions ("" if none); with
    numbers=True, first the digits-of-negative-numbers entry if the task mentions digits."""
    bank = ((NUMBER_ENTRY,) if numbers else ()) + QUESTION_BANK
    lines = [entry for pattern, entry in bank if re.search(pattern, prompt, re.IGNORECASE)][
        :MAX_CHECKLIST
    ]
    if not lines:
        return ""
    return CHECKLIST_HEADER + "\n" + "".join(f"- {line}\n" for line in lines)


DECISION_PATTERN = re.compile(r"DECISION[\s:*`]*(ASK|CODE)", re.IGNORECASE)


QUESTION_PATTERN = re.compile(r"QUESTION\s*:\s*(.+)", re.IGNORECASE)


def atomic_question(question: str | None) -> str | None:
    """Keeps only the first question of a bundle ("...? Also, ...?"): the simulated user
    refuses bundles ("one question at a time") or answers them only vaguely."""
    if not question:
        return None
    first = re.split(r"\s+(?:Also|Additionally|And also|Furthermore)\b[,:]?\s", question)[0]
    if "?" in first:
        first = first[: first.index("?") + 1]
    return first if len(first) >= MIN_QUESTION_LENGTH else question


QUESTION_JUDGE_TEMPLATE = """
Task:
<<<
{prompt}
>>>
Only ONE of these clarifying questions can be asked to the user who wrote the task:
{questions}

Judge each question by five criteria:
- CRITICALITY: its answer changes the values the function must return on ordinary inputs.
- SEARCH SPACE: the task text and its examples do not already settle it.
- LEAKAGE: it asks for a behavioural fact, not for test cases, the algorithm or the code.
- ATOMICITY: a complete answer is exactly one fact: no bundled sub-questions ("... and ...?"),
  nothing so broad that answering it reveals several facts (e.g. "What are the inputs and
  outputs?").
- OBJECTIVITY: it has one unambiguous answer, not a matter of taste.
For each question write one line: <number>: PASS or FAIL, with the criteria it fails. Then
choose, among the questions that pass, the one whose answer would make the most hidden tests
pass (contradictions and the return format first). End with exactly one line: BEST: <number>
""".strip()


BEST_PATTERN = re.compile(r"BEST\W*(\d+)", re.I)


QUESTION_PRIORITY = """
Among the questions that pass, prefer by this order of value:
1. a contradiction between two statements, or between a statement and an example, the
   signature or a type hint;
2. the type or format of the return value;
3. the arguments (which, in what order, of what type) when the task shows no signature;
4. a missing rule or a vague term that changes results on ordinary inputs.
A question about a rare or edge input (accents, empty input, negative numbers), about a
requirement tests cannot observe, or about something a careful programmer would decide the
same way anyway is worth less than any of these.
"""


def _normalized(question: str) -> str:
    return " ".join(re.findall(r"\w+", question.lower()))


def judge_questions(
    env: ClarificationEnvironment, prompt: str, questions: list[str], priority: bool = False
) -> int:
    """Index of the question the judge prefers (0, the first, when it gives no clear verdict)."""
    if len(questions) < 2:
        return 0
    template = QUESTION_JUDGE_TEMPLATE
    if priority:
        template = template.replace(
            "End with exactly one line:", QUESTION_PRIORITY.strip() + "\nEnd with exactly one line:"
        )
    response = env.llm(
        _fill(
            template,
            prompt=prompt,
            questions="\n".join(f"{n}. {q}" for n, q in enumerate(questions, 1)),
        )
    )
    verdicts = BEST_PATTERN.findall(response or "")
    if verdicts and 1 <= int(verdicts[-1]) <= len(questions):
        return int(verdicts[-1]) - 1
    return 0


def _parse_decision(response: str) -> str | None:
    match = DECISION_PATTERN.search(response)
    return match.group(1).upper() if match else None


def _parse_question(response: str) -> str | None:
    match = QUESTION_PATTERN.search(response)
    if not match:
        return None
    question = match.group(1).split("```", 1)[0].strip().strip("*`'\" ").strip()
    if len(question) < MIN_QUESTION_LENGTH:
        return None
    return question[:MAX_QUESTION_LENGTH]


MAX_AMBIGUITIES = 6


DEFECT_LINE = re.compile(r"^\s*[-*]\s*(.+\|.+)$")


def audit_defects(audit: str) -> list[str]:
    """The defect lines the audit wrote in STEP 1 (none when it wrote "- none")."""
    head = re.split(r"STEP\s*2|DECISION", audit, maxsplit=1, flags=re.I)[0]
    lines = [m.group(1).strip() for m in map(DEFECT_LINE.match, head.splitlines()) if m]
    return lines[:MAX_AMBIGUITIES]


# ---------------------------------------------------------------------------------------------
# 2. z3 evidence: stated examples, inferred signature, conflicts proven by the solver
# ---------------------------------------------------------------------------------------------
# extract_examples reads `f(args) == value` and doctest examples from the task; infer_signature
# gives the argument and result sorts; type_clash finds a hint an example violates without a
# solver. find_certificate asks the LLM for SMT-LIB statements and lets z3 prove a statement and
# an example inconsistent (an unsat core). find_interpretation_conflict formalizes the two
# readings of each audit defect and lets z3 find a concrete input where they differ. Every
# function here returns None on any failure: the solver can only add evidence, never break.

MAX_EXAMPLES = 8


MAX_STATEMENTS = 12


MAX_QUOTE_LENGTH = 160


SOLVER_TIMEOUT_MS = 2000  # per z3 check


MAX_SOLVER_CHECKS = 24


SOLVER_BUDGET_S = 10.0  # all z3 work for one task


FORMALIZE_TEMPLATE = """
Below is the task text for a Python function `{entry_point}`. Some of its statements may
contradict each other or its examples. Formalize every statement of the description that
constrains the return value as first-order logic in SMT-LIB 2, over this declaration:

{declaration}

Task:
<<<
{prompt}
>>>

For each such statement write exactly two lines:
STATEMENT "<exact quote from the task>"
ASSERT <one SMT-LIB 2 formula on a single line, using f>

- Use (forall ((x Sort) ...) ...) for what holds for every input, and (=> condition ...) for
  conditional rules. Available: Int/Real arithmetic (+ - * div mod abs to_real to_int), Bool,
  strings (str.len str.at str.substr str.++ str.contains str.prefixof str.suffixof str.indexof
  str.to_int str.from_int) and sequences (seq.len seq.nth seq.unit seq.++ seq.contains
  seq.extract). A helper may be declared once, on its own line: DEFINE (define-fun ...) or
  DEFINE (define-fun-rec ...).
- Formalize each statement literally, even when it seems to conflict with the examples or with
  another statement. Do not resolve conflicts and never invent requirements.
- Do not formalize the examples (`>>>` lines or `f(...) == ...` lines): they are added
  automatically. Skip statements that are vague or that you cannot express.

Example, for `def clip(x: int) -> int` described as "Return x, but never more than 10.":
STATEMENT "Return x, but never more than 10."
ASSERT (forall ((x Int)) (and (<= (f x) 10) (=> (<= x 10) (= (f x) x))))
""".strip()


INLINE_EXAMPLE_OP = re.compile(r"\s*(?:==>|==|➞|=>|->|→|should return|returns|return)\s*", re.I)


SMT_COMMAND = re.compile(
    r"\(\s*(?:assert|check-sat|push|pop|reset|exit|set-|get-|eval|echo|declare|define)", re.I
)


F_SYMBOL = re.compile(r"(?<![\w.$-])f(?![\w.$-])")


SCALAR_SORTS = {"int": "Int", "float": "Real", "bool": "Bool", "str": "String"}


SEQUENCE_NAMES = {"List", "list", "Sequence", "Iterable", "Tuple", "tuple"}


COMMON_WORDS = frozenset(
    "the that this with from function return returns should which given each into only when "
    "then than have will what there their value values list number numbers string strings "
    "element elements write task says also".split()
)


def _balanced_parens(text: str, start: int) -> int | None:
    """Index just past the bracket group opening at text[start], skipping string literals."""
    depth = 0
    quote = None
    i = start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _literal(text: str) -> tuple[bool, Any]:
    """Parses a Python literal; falls back to the longest word-prefix that parses, so trailing
    prose ("3 (since ...)") does not lose an example."""
    words = text.strip().split()
    for k in range(len(words), max(len(words) - 12, 0), -1):
        candidate = " ".join(words[:k]).rstrip(".,;:").strip()
        if not candidate:
            continue
        try:
            return True, ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return False, None


def extract_examples(prompt: str, entry_point: str) -> list[dict[str, Any]]:
    """Examples `entry_point(args) -> value` written in the prompt (doctest or inline style)."""
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = prompt.splitlines()
    call = re.compile(r"(?<![\w.])" + re.escape(entry_point) + r"\s*\(")
    for idx, line in enumerate(lines):
        for match in call.finditer(line):
            end = _balanced_parens(line, match.end() - 1)
            if end is None:
                continue
            inner = line[match.end() : end - 1]
            ok, args = _literal("(" + inner + ",)") if inner.strip() else (True, ())
            if not ok or not isinstance(args, tuple):
                continue
            rest = line[end:]
            op = INLINE_EXAMPLE_OP.match(rest)
            if op:
                ok, expected = _literal(rest[op.end() :])
            elif line.strip().startswith(">>>") and not rest.strip():
                following = [
                    nxt.strip()
                    for nxt in lines[idx + 1 : idx + 4]
                    if nxt.strip() and not nxt.strip().startswith(">>>")
                ]
                ok, expected = _literal(following[0]) if following else (False, None)
            else:
                ok = False
            if not ok:
                continue
            key = repr((args, expected))
            if key in seen:
                continue
            seen.add(key)
            call_text = f"{entry_point}{line[match.end() - 1 : end]}"
            examples.append(
                {
                    "args": args,
                    "expected": expected,
                    "call": call_text,
                    "quote": f"{call_text} == {expected!r}",
                }
            )
            if len(examples) >= MAX_EXAMPLES:
                return examples
    return examples


def stub_annotations(prompt: str, entry_point: str) -> tuple[list[str], list, Any] | None:
    """(parameter names, their annotation nodes, return annotation node) of the `def` stub."""
    match = re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(", prompt)
    if not match:
        return None
    end = _balanced_parens(prompt, match.end() - 1)
    if end is None:
        return None
    returns = re.match(r"\s*->\s*([^:\n]+):", prompt[end:])
    source = f"def f{prompt[match.end() - 1 : end]}"
    source += f" -> {returns.group(1).strip()}: pass" if returns else ": pass"
    try:
        function = ast.parse(source).body[0]
    except SyntaxError:
        return None
    params = [*function.args.posonlyargs, *function.args.args]
    return [p.arg for p in params], [p.annotation for p in params], function.returns


def hint_sort(node: Any) -> str | None:
    """SMT sort for a type hint; '(Seq ?)' for a bare list; None when unknown or unsupported."""
    if isinstance(node, ast.Name):
        if node.id in SCALAR_SORTS:
            return SCALAR_SORTS[node.id]
        return "(Seq ?)" if node.id in SEQUENCE_NAMES else None
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        if base not in SEQUENCE_NAMES:
            return None
        inner = node.slice
        if isinstance(inner, ast.Tuple):  # Tuple[int, ...] / Tuple[int, int]: homogeneous only
            elements = [
                e for e in inner.elts if not (isinstance(e, ast.Constant) and e.value is Ellipsis)
            ]
            sorts = {hint_sort(e) for e in elements}
            return f"(Seq {sorts.pop()})" if len(sorts) == 1 and None not in sorts else None
        element = hint_sort(inner)
        return f"(Seq {element})" if element else None
    return None


def value_sort(value: Any) -> str | None:
    """Most specific sort of a Python value: '(Seq ?)' for an empty list, None if unsupported."""
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        return "Int"
    if isinstance(value, float):
        return "Real"
    if isinstance(value, str):
        return "String"
    if isinstance(value, (list, tuple)):
        element = unify([value_sort(v) for v in value])
        return None if element is None else f"(Seq {element})"
    return None


def unify(sorts: list[str | None]) -> str | None:
    """Join of sorts: Int+Real -> Real; '?' fits anything; None when they clash."""
    if any(s is None for s in sorts):
        return None
    known = [s for s in sorts if s != "?"]
    if not known:
        return "?"
    if all(s in ("Int", "Real") for s in known):
        return "Real" if "Real" in known else "Int"
    if all(s.startswith("(Seq ") for s in known):
        inner = unify([s[5:-1] for s in known])
        return None if inner is None else f"(Seq {inner})"
    return known[0] if all(s == known[0] for s in known) else None


def _complete(sort: str | None) -> bool:
    return sort is not None and "?" not in sort


def _smt_string(value: str) -> str:
    out = []
    for ch in value:
        if ch == '"':
            out.append('""')
        elif 32 <= ord(ch) < 127 and ch != "\\":
            out.append(ch)
        else:
            out.append("\\u{%x}" % ord(ch))
    return '"' + "".join(out) + '"'


def smt_value(value: Any, sort: str) -> str | None:
    """SMT-LIB term for a Python value of the given sort, or None when it does not fit.
    Mirrors Python equality where tests rely on it: 3 == 3.0 and True == 1."""
    if sort == "Bool":
        if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
            return "true" if value else "false"
        return None
    if sort == "Int":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            return None
        value = int(value)
        return str(value) if value >= 0 else f"(- {-value})"
    if sort == "Real":
        if (
            not isinstance(value, (int, float))
            or value != value
            or value
            in (
                float("inf"),
                float("-inf"),
            )
        ):
            return None
        q = Fraction(value)
        term = f"{abs(q.numerator)}.0"
        if q.denominator != 1:
            term = f"(/ {term} {q.denominator}.0)"
        return term if q >= 0 else f"(- {term})"
    if sort == "String":
        return _smt_string(value) if isinstance(value, str) else None
    if sort.startswith("(Seq ") and sort.endswith(")"):
        if not isinstance(value, (list, tuple)):
            return None
        if not value:
            return f"(as seq.empty {sort})"
        parts = [smt_value(v, sort[5:-1]) for v in value]
        if any(p is None for p in parts):
            return None
        units = [f"(seq.unit {p})" for p in parts]
        return units[0] if len(units) == 1 else "(seq.++ " + " ".join(units) + ")"
    return None


def infer_signature(
    prompt: str, entry_point: str, examples: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Argument and result sorts from the type hints, completed by the example values."""
    stub = stub_annotations(prompt, entry_point)
    arity = len(stub[0]) if stub else Counter(len(e["args"]) for e in examples).most_common(1)[0][0]
    names = stub[0] if stub else [f"x{i + 1}" for i in range(arity)]
    arg_hints = stub[1] if stub else [None] * arity
    usable = [e for e in examples if len(e["args"]) == arity]
    if not usable:
        return None

    def resolve(hint_node: Any, values: list[Any]) -> tuple[str | None, str | None]:
        hinted = hint_sort(hint_node)
        if _complete(hinted):
            return hinted, hinted
        inferred = unify([value_sort(v) for v in values])
        if inferred is None:  # clashing example values: keep the most common sort
            counts = Counter(value_sort(v) for v in values if value_sort(v) is not None)
            inferred = counts.most_common(1)[0][0] if counts else None
        if hinted is not None and inferred is not None:
            inferred = unify([hinted, inferred])
        return (inferred if _complete(inferred) else None), None

    arg_sorts, arg_hinted = [], []
    for i in range(arity):
        sort, hinted = resolve(arg_hints[i], [e["args"][i] for e in usable])
        arg_sorts.append(sort)
        arg_hinted.append(hinted)
    result_sort, result_hinted = resolve(stub[2] if stub else None, [e["expected"] for e in usable])
    if result_sort is None or any(s is None for s in arg_sorts):
        return None
    return {
        "names": names,
        "arg_sorts": arg_sorts,
        "result_sort": result_sort,
        "arg_hints": [ast.unparse(h) if h is not None else None for h in arg_hints],
        "result_hint": ast.unparse(stub[2]) if stub and stub[2] is not None else None,
        "arg_hinted": arg_hinted,
        "result_hinted": result_hinted,
        "examples": usable,
    }


def _short(quote: str) -> str:
    quote = re.sub(r"\s+", " ", quote).strip()
    return quote if len(quote) <= MAX_QUOTE_LENGTH else quote[: MAX_QUOTE_LENGTH - 3] + "..."


def type_clash(signature: dict[str, Any], entry_point: str) -> dict[str, Any] | None:
    """A type hint that an example violates: `-> bool` but the example returns 'False'."""
    for example in signature["examples"]:
        value, hinted = example["expected"], signature["result_hinted"]
        if hinted and value_sort(value) and smt_value(value, hinted) is None:
            question = (
                f"The signature of `{entry_point}` says it returns `{signature['result_hint']}`, "
                f'but the example "{_short(example["quote"])}" returns {value!r}. '
                "Which is intended?"
            )
            quotes = (f"-> {signature['result_hint']}", example["quote"])
            return {"kind": "type", "question": question, "quotes": quotes}
        for name, hint, hinted, value in zip(
            signature["names"],
            signature["arg_hints"],
            signature["arg_hinted"],
            example["args"],
            strict=True,
        ):
            if hinted and value_sort(value) and smt_value(value, hinted) is None:
                question = (
                    f"The signature of `{entry_point}` says `{name}` is `{hint}`, but the "
                    f'example "{_short(example["quote"])}" passes {value!r}. Which is intended?'
                )
                return {"kind": "type", "question": question, "quotes": (hint, example["quote"])}
    return None


def parse_smt_statements(response: str, prompt: str) -> tuple[list[str], list[dict[str, Any]]]:
    """(DEFINE lines, [{quote, formulas}]) from the response; unverifiable quotes are dropped."""
    defines: list[str] = []
    statements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in response.splitlines():
        line = re.sub(r"^[\s\-*•>#\d.)]*", "", raw.replace("**", "")).strip().strip("`").strip()
        statement = re.match(r"^STATEMENT\s*:?\s*(.+)$", line, re.I)
        if statement:
            quote = statement.group(1).strip().strip("“”\"'").strip()
            current = None
            if quote and quote_in_text(quote, prompt) and len(statements) < MAX_STATEMENTS:
                current = {"quote": quote[:MAX_QUOTE_LENGTH], "formulas": []}
                statements.append(current)
            continue
        formula = re.match(r"^ASSERT\s*:?\s*(\(.+\))\s*$", line, re.I)
        if formula and current is not None:
            current["formulas"].append(formula.group(1))
            continue
        define = re.match(r"^DEFINE\s*:?\s*(\(\s*define-fun(?:-rec)?\b.+\))\s*$", line, re.I)
        if define:
            defines.append(define.group(1))
    return defines, [s for s in statements if s["formulas"]]


def _usable_formula(formula: str) -> str | None:
    """The bare formula, or None if it is not a single command-free term mentioning f."""
    wrapped = re.match(r"^\(\s*assert\s+(.+)\)$", formula.strip(), re.S)
    body = wrapped.group(1).strip() if wrapped else formula.strip()
    if body.count("(") != body.count(")") or SMT_COMMAND.search(body) or not F_SYMBOL.search(body):
        return None
    return body


def _render(value: Any) -> Any:
    """A z3 value as a Python literal, or None when it has no simple Python form."""
    if z3.is_true(value):
        return True
    if z3.is_false(value):
        return False
    if z3.is_int_value(value):
        return value.as_long()
    if z3.is_rational_value(value):
        fraction = value.as_fraction()
        as_float = float(fraction)
        return as_float if Fraction(as_float) == fraction else None
    if z3.is_string_value(value):
        return value.as_string()
    return None


def smt_certificate(
    signature: dict[str, Any],
    defines: list[str],
    statements: list[dict[str, Any]],
    entry_point: str,
) -> dict[str, Any] | None:
    """Proves a contradiction between the formalized statements and the prompt's examples."""
    declaration = f"(declare-fun f ({' '.join(signature['arg_sorts'])}) {signature['result_sort']})"

    def parses(script: str) -> bool:
        # A throwaway context per check: z3 registers a define-fun-rec in its context on the
        # first parse, so parsing the same helper twice in one context fails.
        try:
            z3.parse_smt2_string(script, ctx=z3.Context())
            return True
        except z3.Z3Exception:
            return False

    prefix = declaration
    for define in defines:  # each helper must parse on top of the previous ones
        if parses(prefix + define):
            prefix += define

    # Pieces that parse on their own, in order; the whole theory is then parsed exactly once.
    pieces: list[tuple[int, str]] = []  # (group index, assertion text)
    groups: list[dict[str, Any]] = []
    for statement in statements:
        bodies = [_usable_formula(formula) for formula in statement["formulas"]]
        bodies = [b for b in bodies if b is not None and parses(f"{prefix}(assert {b})")]
        if bodies:
            pieces.extend((len(groups), f"(assert {b})") for b in bodies)
            groups.append({"kind": "statement", "quote": statement["quote"], "exprs": []})
    for example in signature["examples"]:
        args = [
            smt_value(a, s) for a, s in zip(example["args"], signature["arg_sorts"], strict=True)
        ]
        result = smt_value(example["expected"], signature["result_sort"])
        if result is None or any(a is None for a in args):
            continue
        call = f"(f {' '.join(args)})" if args else "f"
        if parses(f"{prefix}(assert (= {call} {result}))"):
            pieces.append((len(groups), f"(assert (= {call} {result}))"))
            groups.append({"kind": "example", "quote": example["quote"], "exprs": []})
    if not any(g["kind"] == "statement" for g in groups) and len(groups) < 2:
        return None

    ctx = z3.Context()
    try:
        assertions = z3.parse_smt2_string(prefix + "".join(text for _, text in pieces), ctx=ctx)
    except z3.Z3Exception:
        return None
    if len(assertions) != len(pieces):
        return None
    for (index, _), expr in zip(pieces, assertions, strict=True):
        groups[index]["exprs"].append(expr)
    calls = {g["quote"]: g["exprs"][0].arg(0) for g in groups if g["kind"] == "example"}

    checks = [0]
    deadline = time.monotonic() + SOLVER_BUDGET_S

    def check(solver: Any, *assumptions: Any) -> Any:
        checks[0] += 1
        if checks[0] > MAX_SOLVER_CHECKS or time.monotonic() > deadline:
            return z3.unknown  # out of budget: treated as "not proven"
        return solver.check(*assumptions)

    def new_solver() -> Any:
        solver = z3.Solver(ctx=ctx)
        solver.set("timeout", SOLVER_TIMEOUT_MS)
        return solver

    for group in groups:  # a statement unsatisfiable on its own is a formalization error
        if group["kind"] == "statement":
            alone = new_solver()
            alone.add(*group["exprs"])
            if check(alone) == z3.unsat:
                group["exprs"] = []
    groups = [g for g in groups if g["exprs"]]
    if not any(g["kind"] == "statement" for g in groups) and len(groups) < 2:
        return None

    solver = new_solver()
    trackers = []
    for i, group in enumerate(groups):
        tracker = z3.Bool(f"group{i}", ctx)
        for expr in group["exprs"]:
            solver.add(z3.Implies(tracker, expr))
        trackers.append(tracker)
    if check(solver, *trackers) != z3.unsat:
        return None
    named = {str(t) for t in solver.unsat_core()}
    core = [i for i in range(len(groups)) if f"group{i}" in named]
    i = 0
    while i < len(core):  # deletion-based minimisation; "unknown" keeps the group
        trial = core[:i] + core[i + 1 :]
        if trial and check(solver, *[trackers[j] for j in trial]) == z3.unsat:
            core = trial
        else:
            i += 1

    found = [groups[i] for i in core]
    examples = [g for g in found if g["kind"] == "example"]
    prose = [g for g in found if g["kind"] == "statement"]
    if not examples:
        return None  # prose-only conflicts depend entirely on the LLM's formalization
    example = examples[0]
    if not prose:
        if len(examples) < 2:
            return None
        question = (
            f'The examples "{_short(examples[0]["quote"])}" and "{_short(examples[1]["quote"])}" '
            "give different results for the same input. Which one is intended?"
        )
        return {
            "kind": "examples",
            "question": question,
            "quotes": (examples[0]["quote"], examples[1]["quote"]),
        }

    implied = None
    call_term = calls.get(example["quote"])
    if call_term is not None:
        implication = new_solver()
        implication.add(*[e for g in prose for e in g["exprs"]])
        if check(implication) == z3.sat:
            value = implication.model().eval(call_term, model_completion=True)
            implication.add(call_term != value)
            if check(implication) == z3.unsat:  # the statements force exactly this value
                implied = _render(value)
    call_text = example["quote"].split(" == ", 1)[0]
    statement = _short(prose[0]["quote"])
    if implied is not None:
        question = (
            f'The task says "{statement}", which gives `{call_text} == {implied!r}`, but the '
            f'example says "{_short(example["quote"])}". Which is intended?'
        )
    else:
        question = (
            f'The task says "{statement}", but the example "{_short(example["quote"])}" does not '
            f"satisfy it. Which one should `{entry_point}` follow?"
        )
    return {
        "kind": "smt",
        "question": question[:MAX_QUESTION_LENGTH],
        "quotes": (prose[0]["quote"], example["quote"]),
        "implied": implied,
    }


def find_certificate(
    env: ClarificationEnvironment, problem: dict[str, Any], solver: bool = True
) -> dict[str, Any] | None:
    """The whole contradiction check. Never raises: any failure means "no certificate".
    solver=False behaves as if z3 were not installed (only the type clash, which needs none)."""
    prompt, entry_point = problem["prompt"], problem["entry_point"]
    try:
        examples = extract_examples(prompt, entry_point)
        if not examples:
            return None
        signature = infer_signature(prompt, entry_point, examples)
        if signature is None:
            return None
        clash = type_clash(signature, entry_point)
        if clash is not None or z3 is None or not solver:
            return clash
        params = ", ".join(
            f"{name} : {sort}"
            for name, sort in zip(signature["names"], signature["arg_sorts"], strict=True)
        )
        declaration = (
            f"(declare-fun f ({' '.join(signature['arg_sorts'])}) {signature['result_sort']})\n"
            f"; f({', '.join(signature['names'])}) stands for {entry_point}; {params}; "
            f"result : {signature['result_sort']}"
        )
        response = env.llm(
            _fill(
                FORMALIZE_TEMPLATE, prompt=prompt, entry_point=entry_point, declaration=declaration
            )
        )
        defines, statements = parse_smt_statements(response, prompt)
        if not statements:
            return None
        return smt_certificate(signature, defines, statements, entry_point)
    except Exception:
        return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote's word tokens occur contiguously in the text (case/punctuation-blind)."""
    needle, hay = _tokens(quote), _tokens(text)
    if not needle:
        return False
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def _stems(text: str) -> set[str]:
    # A crude 5-letter stem: "subtract" and "subtraction" must count as the same word.
    return {w[:5] for w in _tokens(text) if len(w) >= 4 and w not in COMMON_WORDS}


RANDOM_INPUTS = 12


RANDOM_ALPHABET = "abcAB 1"


SUPPORTED_SORT = re.compile(
    r"^(?:Int|Real|Bool|String|\(Seq (?:Int|Real|Bool|String|\(Seq (?:Int|Real|Bool|String)\))\))$"
)


DEFINE_COMMAND = re.compile(
    r"\(\s*(?:define|declare|assert|check-sat|push|pop|reset|exit|set-|get-|eval|echo)", re.I
)


READINGS_TEMPLATE = """
Below is a task for a Python function `{entry_point}`, and the places where an audit found that
the task can be read in two ways. Formalize both readings of each place as SMT-LIB 2 function
definitions, so that a solver can search for an input on which they return different values.

Task:
<<<
{prompt}
>>>

Places found by the audit:
{defects}

For each place you can formalize, write exactly these five lines:
AMBIGUITY "<exact phrase from the task, or: not stated>"
READING A: <one short plain-English clause>
DEFINE A (define-fun-rec fa ((x1 Sort) ...) Sort body)
READING B: <one short plain-English clause>
DEFINE B (define-fun-rec fb ((x1 Sort) ...) Sort body)

- fa and fb each compute the WHOLE result of `{entry_point}` under that reading. They take the
  same parameters, in the same order, with the same sorts, and differ only at this place.
- Sorts: Int, Real, Bool, String, (Seq Int), (Seq Real), (Seq String), (Seq (Seq Int)).
- Use arithmetic (+ - * div mod abs), ite, and, or, not, =, str.* and seq.* functions, and loop
  over a sequence or a string by recursion (a define-fun-rec calling itself on the rest).
  A helper shared by several definitions may be written once, on its own line:
  HELPER (define-fun-rec name ((x Sort) ...) Sort body)
- Every DEFINE and HELPER fits on a single line. Skip a place you cannot formalize, and skip
  places about the parameters themselves (different parameter lists cannot be compared).
{parameters}
""".strip()


def phrase_overlap(phrase: str, text: str, threshold: float = 0.6) -> bool:
    """Lenient check used only for wording a question: an exact quote, or a near-quote most of
    whose words appear in the text."""
    if quote_in_text(phrase, text):
        return True
    words = set(_tokens(phrase))
    return len(words) >= 3 and len(words & set(_tokens(text))) / len(words) >= threshold


def _clean_reading(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip("`'\"").strip().rstrip(" .;?!")
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        text = text[0].lower() + text[1:]
    return text


def _sexpr(text: str) -> list | None:
    """One s-expression as nested lists of atoms (string literals keep their quotes)."""
    stack: list[list] = [[]]
    for token in re.findall(r'"(?:[^"]|"")*"|\(|\)|[^\s()"]+', text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) == 1:
                return None
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    if len(stack) != 1 or len(stack[0]) != 1 or not isinstance(stack[0][0], list):
        return None
    return stack[0][0]


def _sort_text(node: Any) -> str:
    return node if isinstance(node, str) else "(" + " ".join(_sort_text(n) for n in node) + ")"


def definition_signature(define: str) -> tuple[str, list[str], str] | None:
    """(name, parameter sorts, result sort) of one define-fun / define-fun-rec command."""
    if define.count("(") != define.count(")") or len(DEFINE_COMMAND.findall(define)) != 1:
        return None
    tree = _sexpr(define)
    if not tree or len(tree) != 5 or tree[0] not in ("define-fun", "define-fun-rec"):
        return None
    name, params = tree[1], tree[2]
    if not isinstance(name, str) or not isinstance(params, list):
        return None
    sorts = []
    for param in params:
        if not (isinstance(param, list) and len(param) == 2 and isinstance(param[0], str)):
            return None
        sorts.append(_sort_text(param[1]))
    return name, sorts, _sort_text(tree[3])


def _renamed(define: str, old: str, new: str) -> str:
    # z3 inlines define-fun as a macro; define-fun-rec keeps calls as applications, which is
    # what lets the witness and the values be read back from the model.
    define = re.sub(r"^\(\s*define-fun\s", "(define-fun-rec ", define.strip())
    return re.sub(r"(?<![\w.$-])" + re.escape(old) + r"(?![\w.$-])", new, define)


def parse_readings(response: str, prompt: str) -> tuple[list[str], list[dict[str, Any]]]:
    """(HELPER definitions, complete ambiguities with both readings and both definitions)."""
    helpers: list[str] = []
    ambiguities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in response.splitlines():
        line = re.sub(r"^[\s\-*•>#\d.)]*", "", raw.replace("**", "")).strip().strip("`").strip()
        ambiguity = re.match(r"^AMBIGUITY\s*:?\s*(.*)$", line, re.I)
        if ambiguity:
            quote = ambiguity.group(1).strip().strip("“”\"'").strip()
            if quote.lower().startswith("not stated") or not phrase_overlap(quote, prompt):
                quote = ""
            current = {
                "quote": quote[:MAX_QUOTE_LENGTH],
                "A": "",
                "B": "",
                "def_A": "",
                "def_B": "",
            }
            if len(ambiguities) < MAX_AMBIGUITIES:
                ambiguities.append(current)
            continue
        reading = re.match(r"^READING\s+([AB])\s*[:)]\s*(.+)$", line, re.I)
        if reading and current is not None:
            current[reading.group(1).upper()] = _clean_reading(reading.group(2))
            continue
        define = re.match(r"^DEFINE\s+([AB])\s*:?\s*(\(\s*define-fun.*\))\s*$", line, re.I)
        if define and current is not None:
            current["def_" + define.group(1).upper()] = define.group(2)
            continue
        helper = re.match(r"^HELPER\s*:?\s*(\(\s*define-fun.*\))\s*$", line, re.I)
        if helper:
            helpers.append(helper.group(1))
    complete = [a for a in ambiguities if a["A"] and a["B"] and a["def_A"] and a["def_B"]]
    return helpers, complete


def _render_value(value: Any) -> Any:
    """Like _render, plus (nested) sequences."""
    if z3.is_seq(value) and not z3.is_string_value(value):
        length = z3.simplify(z3.Length(value))
        if not z3.is_int_value(length):
            return None
        items = [_render_value(z3.simplify(value[i])) for i in range(length.as_long())]
        return None if any(item is None for item in items) else items
    return _render(value)


def _bounds(term: str, sort: str, ordinary: bool) -> list[str]:
    """Small-input bounds; "ordinary" also excludes empty and negative inputs."""
    if sort == "Int":
        return [f"(<= {'0' if ordinary else '(- 10)'} {term} 10)"]
    if sort == "Real":
        return [f"(<= {'0.0' if ordinary else '(- 10.0)'} {term} 10.0)"]
    if sort == "String":
        return [f"(<= {1 if ordinary else 0} (str.len {term}) 5)"]
    if sort.startswith("(Seq "):
        bounds = [f"(<= {1 if ordinary else 0} (seq.len {term}) 4)"]
        inner = _bounds(f"(seq.nth {term} i)", sort[5:-1], ordinary)
        if inner and not sort[5:-1].startswith("(Seq"):
            bounds.append(
                f"(forall ((i Int)) (=> (and (<= 0 i) (< i (seq.len {term}))) {inner[0]}))"
            )
        return bounds
    return []


def _random_value(sort: str, rng: random.Random) -> Any:
    if sort == "Int":
        return rng.randint(0, 10)
    if sort == "Real":
        return rng.choice([0.5, 1.0, 2.5, 3.0, 7.25])
    if sort == "Bool":
        return rng.random() < 0.5
    if sort == "String":
        return "".join(rng.choice(RANDOM_ALPHABET) for _ in range(rng.randint(1, 5)))
    if sort.startswith("(Seq "):
        return [_random_value(sort[5:-1], rng) for _ in range(rng.randint(1, 4))]
    return None


def analyse_ambiguity(
    prefix: str, name_a: str, name_b: str, sorts: list[str], result_sort: str, budget: Any
) -> dict[str, Any] | None:
    """Witness search (ordinary inputs first) and divergence rate for two parsed readings."""
    consts = [f"input{i}" for i in range(len(sorts))]
    declarations = "".join(f"(declare-const {c} {s})" for c, s in zip(consts, sorts, strict=True))
    call_a = f"({name_a} {' '.join(consts)})" if consts else name_a
    call_b = f"({name_b} {' '.join(consts)})" if consts else name_b
    analysis: dict[str, Any] = {"witness": None, "ordinary": False, "rate": None}

    proven_equal = True
    for ordinary in (True, False):
        bounds = [b for c, s in zip(consts, sorts, strict=True) for b in _bounds(c, s, ordinary)]
        script = (
            prefix
            + declarations
            + "".join(f"(assert {b})" for b in bounds)
            + "".join(f"(assert (= {c} {c}))" for c in consts)
            + f"(assert (not (= {call_a} {call_b})))"
        )
        ctx = z3.Context()
        try:
            assertions = z3.parse_smt2_string(script, ctx=ctx)
        except z3.Z3Exception:
            return None
        solver = z3.Solver(ctx=ctx)
        solver.set("timeout", SOLVER_TIMEOUT_MS)
        solver.add(*assertions)
        result = budget(solver)
        if result == z3.unsat:
            continue
        proven_equal = False
        if result != z3.sat:
            continue
        model = solver.model()
        start = len(bounds)
        args = [
            _render_value(model.eval(assertions[start + i].arg(0), model_completion=True))
            for i in range(len(consts))
        ]
        differ = assertions[len(assertions) - 1].arg(0)
        values = [_render_value(model.eval(differ.arg(j), model_completion=True)) for j in (0, 1)]
        if any(a is None for a in args) or any(v is None for v in values) or values[0] == values[1]:
            continue
        analysis.update(witness=(args, values[0], values[1]), ordinary=ordinary)
        break
    if proven_equal:
        analysis["equivalent"] = True
        return analysis

    rng = random.Random(len(prefix))  # deterministic per formalization
    inputs = [[_random_value(s, rng) for s in sorts] for _ in range(RANDOM_INPUTS)]
    lines = []
    for i, values in enumerate(inputs):
        encoded = [smt_value(v, s) for v, s in zip(values, sorts, strict=True)]
        if any(e is None for e in encoded):
            return analysis
        args = " ".join(encoded)
        lines.append(
            f"(declare-const out_a{i} {result_sort})(declare-const out_b{i} {result_sort})"
            f"(assert (= out_a{i} ({name_a} {args})))(assert (= out_b{i} ({name_b} {args})))"
            if args
            else f"(declare-const out_a{i} {result_sort})(declare-const out_b{i} {result_sort})"
            f"(assert (= out_a{i} {name_a}))(assert (= out_b{i} {name_b}))"
        )
    ctx = z3.Context()
    try:
        assertions = z3.parse_smt2_string(prefix + "".join(lines), ctx=ctx)
    except z3.Z3Exception:
        return analysis
    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", SOLVER_TIMEOUT_MS)
    solver.add(*assertions)
    if budget(solver) == z3.sat:
        model = solver.model()
        pairs = [
            (
                _render_value(model.eval(assertions[2 * i].arg(0), model_completion=True)),
                _render_value(model.eval(assertions[2 * i + 1].arg(0), model_completion=True)),
            )
            for i in range(len(inputs))
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if len(pairs) >= len(inputs) // 2:
            analysis["rate"] = sum(a != b for a, b in pairs) / len(pairs)
    return analysis


def analyse_readings(
    helpers: list[str], ambiguities: list[dict[str, Any]], parameter_sorts: list[str] | None
) -> list[dict[str, Any]]:
    """Every formalized ambiguity with its witness / divergence rate / equivalence verdict."""
    deadline = time.monotonic() + SOLVER_BUDGET_S
    checks = [0]

    def budget(solver: Any) -> Any:
        checks[0] += 1
        if checks[0] > MAX_SOLVER_CHECKS or time.monotonic() > deadline:
            return z3.unknown
        return solver.check()

    def parses(script: str) -> bool:
        try:
            z3.parse_smt2_string(script, ctx=z3.Context())
            return True
        except z3.Z3Exception:
            return False

    prefix = ""
    for helper in helpers:
        signature = definition_signature(helper)
        if signature and parses(prefix + _renamed(helper, signature[0], signature[0])):
            prefix += _renamed(helper, signature[0], signature[0])

    results = []
    for k, ambiguity in enumerate(ambiguities):
        sig_a = definition_signature(ambiguity["def_A"])
        sig_b = definition_signature(ambiguity["def_B"])
        if not sig_a or not sig_b or sig_a[1:] != sig_b[1:]:
            continue  # unparseable, or different parameter lists: not comparable
        sorts, result_sort = sig_a[1], sig_a[2]
        if not all(SUPPORTED_SORT.match(s) for s in [*sorts, result_sort]):
            continue
        if parameter_sorts is not None and sorts != parameter_sorts:
            continue
        name_a, name_b = f"reading_a{k}", f"reading_b{k}"
        definitions = _renamed(ambiguity["def_A"], sig_a[0], name_a) + _renamed(
            ambiguity["def_B"], sig_b[0], name_b
        )
        if not parses(prefix + definitions):
            continue
        analysis = analyse_ambiguity(
            prefix + definitions, name_a, name_b, sorts, result_sort, budget
        )
        if analysis is not None:
            results.append({**ambiguity, "order": k, **analysis})
    return results


def _overlap(question: str, text: str) -> float:
    content = _stems(text)
    return len(content & _stems(question)) / len(content) if content else 0.0


def choose_ambiguity(
    results: list[dict[str, Any]], audit_question: str | None
) -> dict[str, Any] | None:
    """The ambiguity to ask about, or None to keep the audit's own question.

    Strength = (diverges on ordinary inputs, divergence rate). The audit's target is replaced
    only when the solver can compare it: it was formalized and another ambiguity is stronger or
    it turned out equivalent. An audit question the solver cannot map (e.g. about the
    parameters) is kept."""
    diverging = [r for r in results if r["witness"]]
    if not diverging:
        return None

    def strength(r: dict[str, Any]) -> tuple:
        return (r["ordinary"], r["rate"] or 0.0, -r["order"])

    best = max(diverging, key=strength)
    if audit_question is None:
        return best if best["ordinary"] else None
    scored = [
        (max(_overlap(audit_question, r[key]) for key in ("quote", "A", "B")), r) for r in results
    ]
    score, target = max(scored, key=lambda pair: pair[0], default=(0.0, None))
    if target is None or score < 0.5:
        return None
    if target["witness"] and strength(target)[:2] >= strength(best)[:2]:
        return target  # same target as the audit, asked with a concrete input
    return best


def interpretation_question(result: dict[str, Any], entry_point: str) -> str:
    args, value_a, value_b = result["witness"]
    call = f"{entry_point}({', '.join(repr(a) for a in args)})"
    lead = f'The task says "{_short(result["quote"])}". For example, s' if result["quote"] else "S"
    question = (
        f"{lead}hould `{call}` return {value_a!r} ({result['A']}) or {value_b!r} ({result['B']})?"
    )
    return question[:MAX_QUESTION_LENGTH]


def formalize_readings(
    env: ClarificationEnvironment, problem: dict[str, Any], defects: list[str]
) -> tuple[list[str], list[dict[str, Any]], list[str] | None]:
    """(HELPER definitions, formalized ambiguities, parameter sorts or None) for the audit's
    defects: one READINGS_TEMPLATE call. Raises on LLM failure; callers guard it."""
    prompt, entry_point = problem["prompt"], problem["entry_point"]
    parameter_sorts = None
    examples = extract_examples(prompt, entry_point)
    signature = infer_signature(prompt, entry_point, examples) if examples else None
    stub = stub_annotations(prompt, entry_point)
    if signature is not None:
        parameter_sorts = signature["arg_sorts"]
    elif stub is not None and stub[1] and all(_complete(hint_sort(h)) for h in stub[1]):
        parameter_sorts = [hint_sort(h) for h in stub[1]]
    parameters = ""
    names = signature["names"] if signature is not None else (stub[0] if stub else None)
    if parameter_sorts is not None and names is not None:
        parameters = "- The parameters are, in order: " + ", ".join(
            f"{n} {s}" for n, s in zip(names, parameter_sorts, strict=True)
        )
    response = env.llm(
        _fill(
            READINGS_TEMPLATE,
            prompt=prompt,
            entry_point=entry_point,
            defects="\n".join(f"- {d}" for d in defects),
            parameters=parameters,
        )
    )
    helpers, ambiguities = parse_readings(response, prompt)
    return helpers, ambiguities, parameter_sorts


def find_interpretation_conflict(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    audit: str,
    audit_question: str | None,
    solver: bool = True,
) -> dict[str, Any] | None:
    """The whole interpretation step. Never raises: any failure means "keep the audit's"."""
    try:
        defects = audit_defects(audit)
        if z3 is None or not solver or not defects:
            return None
        helpers, ambiguities, parameter_sorts = formalize_readings(env, problem, defects)
        results = analyse_readings(helpers, ambiguities, parameter_sorts)
        chosen = choose_ambiguity(results, audit_question)
        if chosen is None:
            return None
        return {**chosen, "question": interpretation_question(chosen, problem["entry_point"])}
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# 3. Contract question: a draft's contract for the user to correct
# ---------------------------------------------------------------------------------------------
# CONTRACT_TEMPLATE shows the LLM a draft written without clarification, the audit's defects
# and the evidence (z3's proven conflict, and with several drafts what running them showed),
# and asks for one question that states the draft's contract and asks the user to correct it.

CONTRACT_TEMPLATE = """You wrote this draft of `{entry_point}` for the Python task below, before asking the user
anything. Hidden tests will call `{entry_point}` and compare its return values with what the user
really intends; the task text may be vague, incomplete or contradictory.

Task:
<<<
{prompt}
>>>

Draft:
```python
{draft}
```

Open points you found in the task:
{defects}
{evidence}
You may ask the user ONE question. Use it to let the user correct the draft where it is most likely
wrong. If the task text and one of its examples disagree, ask only about that example: quote the
call with its stated result, give the result the text implies for the same call, and ask which
one is intended. Otherwise state concretely what the draft assumes on ordinary inputs: the
arguments of {entry_point} (types and shape), what it returns (exact type and format) and the rule
it computes, with one small example call and its result. Then ask whether that is right, or
whether it should instead <the most plausible alternative for the open point you are least sure
about>.{order} Start with the part the task leaves most open (the result format, the arguments or the
core rule); mention empty, zero or invalid inputs only if nothing else is open. Never ask for code
or tests. Plain text on one line, no backticks.

Answer with exactly one line:
QUESTION: <your question>"""


SOLVER_EVIDENCE = """
Proven with a solver on the task text (not a guess):
{conflicts}
Your question must settle this conflict first (quote both sides, and the concrete call where
they differ), then state the rest of the draft's contract.
"""


ORDER_RULE = """ The task shows no signature, so name the arguments in their order explicitly
("takes <first argument> first and <second argument> second") so that the user can correct the
order."""


DRAFT_EVIDENCE = """
Observed by running {count} independent drafts in a sandbox (the draft above is the one that
satisfies most stated examples):
{lines}
Ask about what these observations leave open: a stated example no draft satisfies may contradict
the text (ask which is intended), and a call on which the drafts disagree is a point the task does
not fix. Show the competing results for that call in your question.
"""


MAX_DISAGREEMENTS = 3


def draft_evidence(
    env: ClarificationEnvironment, prompt: str, entry_point: str, codes: list[str]
) -> tuple[str, str]:
    """(the draft to ask about, DRAFT_EVIDENCE) from one sandbox run of every draft on the
    stated examples and on probe inputs; (codes[0], "") when nothing could be run."""
    examples = extract_examples(prompt, entry_point)
    probes = []
    try:
        response = env.llm(
            _fill(PROBES_TEMPLATE, prompt=prompt, clarification="", entry_point=entry_point)
        )
        probes = [args for _, args, _ in parse_probe_calls(response, entry_point)][:MAX_PROBES]
    except Exception:
        pass
    calls = [example["args"] for example in examples] + probes
    reports = run_candidates(env, codes, entry_point, calls) if calls else None
    if not reports:
        return codes[0], ""
    usable = [i for i, r in enumerate(reports) if r is not None and not r["load_error"]]
    if not usable:
        return codes[0], ""

    def satisfies(i: int, j: int) -> bool:
        found, value = _result(*reports[i]["results"][j])
        return found and same_value(value, examples[j]["expected"])

    scores = {i: sum(satisfies(i, j) for j in range(len(examples))) for i in usable}
    best = max(usable, key=lambda i: (scores[i], -i))
    lines = []
    for j, example in enumerate(examples):
        if not any(satisfies(i, j) for i in usable):
            got = reports[best]["results"][j][1][:80]
            lines.append(f"- no draft satisfies the stated example {example['quote']}; got {got}")
    for k, args in enumerate(probes):
        index = len(examples) + k
        outcomes: dict[str, int] = {}
        for i in usable:
            kind, info = reports[i]["results"][index]
            shown = info[:80] if kind in ("VALUE", "NONE") else f"raises {info.split(':', 1)[0]}"
            outcomes[shown] = outcomes.get(shown, 0) + 1
        if len(outcomes) > 1 and len(lines) < MAX_DISAGREEMENTS + len(examples):
            call = f"{entry_point}({', '.join(repr(a) for a in args)})"
            results = "; ".join(f"{o} ({n} draft{'s' * (n > 1)})" for o, n in outcomes.items())
            lines.append(f"- the drafts disagree on {call}: {results}")
    if not lines:
        lines = ["- every draft satisfies the stated examples and they agree on every probe"]
    observed = _fill(DRAFT_EVIDENCE, count=str(len(usable)), lines="\n".join(lines))
    return codes[best], observed


def z3_evidence(
    env: ClarificationEnvironment, problem: dict[str, Any], audit: str, solver: bool
) -> str:
    """SOLVER_EVIDENCE for the conflict proven on the task ("" when there is none): first a
    statement or type hint that a stated example violates (find_certificate), else two readings
    of an audit defect that return different values on a concrete input
    (find_interpretation_conflict)."""
    proven = find_certificate(env, problem, solver=solver)
    if proven is None:
        proven = find_interpretation_conflict(env, problem, audit, None, solver)
    if proven is None:
        return ""
    return _fill(SOLVER_EVIDENCE, conflicts="- " + proven["question"].replace("`", ""))


def contract_question(
    env: ClarificationEnvironment,
    prompt: str,
    entry_point: str,
    audit: str,
    rules: str,
    evidence: str = "",
    drafts: int = 1,
    order: bool = False,
) -> str | None:
    """The question stating a draft's contract for the user to correct (None on any failure)."""
    try:
        codes = []
        for _ in range(max(1, drafts)):
            response = env.llm(_fill(CODE_TEMPLATE, prompt=prompt, clarification="", rules=rules))
            code = _check_code(response, entry_point)[0]
            if code:
                codes.append(code)
        if not codes:
            return None
        draft, observed = codes[0], ""
        if len(codes) > 1:
            draft, observed = draft_evidence(env, prompt, entry_point, codes)
        defects = "\n".join(f"- {line}" for line in audit_defects(audit)) or "- none listed"
        shows_def = re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(", prompt)
        reply = env.llm(
            _fill(
                CONTRACT_TEMPLATE,
                prompt=prompt,
                entry_point=entry_point,
                draft=draft,
                defects=defects,
                evidence=observed + evidence,
                order=ORDER_RULE.replace("\n", " ").rstrip() if order and not shows_def else "",
            )
        )
    except Exception:
        return None
    question = _parse_question(reply)
    return question.replace("`", "") if question else None


# ---------------------------------------------------------------------------------------------
# 4. Code with the answer: checks, repairs, finalization
# ---------------------------------------------------------------------------------------------
# _check_code accepts a python block that defines the entry point and is not a placeholder or a
# constant body; _finalize repairs up to MAX_REPAIR_ATTEMPTS times and aliases a renamed entry
# point. usable_answer rejects refusals and dodges (NON_ANSWER).

CODE_TEMPLATE = """
Implement the following Python task.

Task:
<<<
{prompt}
>>>
{clarification}
{rules}
""".strip()


CLARIFICATION_BLOCK = """
You asked the user a clarifying question. Their answer is authoritative: where it conflicts with
the task text, follow the answer.
Question: {question}
Answer: {answer}
"""


NO_ANSWER_BLOCK = """
You asked the user a clarifying question but got no usable answer. Decide every unclear point
yourself: choose the most logical, standard behaviour for a function with this name and purpose.
"""


FALLBACK_TEMPLATE = """
Implement the following Python task. Earlier attempts did not produce a working function, so do
not wait for more information: decide every unclear point yourself, choosing the most logical,
standard behaviour for a function with this name and purpose.

Task:
<<<
{prompt}
>>>

Write a complete, working implementation. A placeholder (`pass`, a bare `return`, `...`, or
raising NotImplementedError) is never acceptable.
{rules}
""".strip()


NON_ANSWER = re.compile(
    r"one (?:question|thing|clarification|fact|aspect) at a time|\bbundl|not atomic"
    r"|\bone specific aspect\b|\bseparate (?:questions|clarifications)\b"
    r"|\bask(?:ed|ing)? (?:\w+ ){0,3}separately|own confirmation"
    r"|\bI (?:can(?:no|')?t|am unable to|'m unable to|won't) "
    r"(?:answer|specify|confirm|say|tell|determine|provide|give|reveal|share|disclose)"
    r"|not (?:fully |entirely |completely )?clear (?:from|in|based on) (?:your|the) question"
    r"|irrelevant question|cannot answer the given question|without more (?:context|detail)",
    re.I,
)


REPAIR_TEMPLATE = """
Your previous answer to the task below cannot be used: {problem}

Task:
<<<
{prompt}
>>>
{clarification}
Previous answer:
<<<
{response}
>>>

Return the corrected, complete implementation of `{entry_point}` in a single ```python block.
""".strip()


def _clarification_block(question: str | None, answer: str | None) -> str:
    if not question or not answer:
        return ""
    return _fill(CLARIFICATION_BLOCK, question=question, answer=answer)


def usable_answer(answer: str | None) -> bool:
    """False for an empty reply or one that refuses or dodges the question."""
    return bool(answer and answer.strip()) and not NON_ANSWER.search(answer)


def _check_code(
    response: str, entry_point: str, degenerate: bool = True
) -> tuple[str | None, str | None]:
    """Returns (code, problem); problem is None when the code is usable as-is.
    degenerate=False skips the empty/constant-body check: exactly Pinpoint's version."""
    try:
        code = _validate_and_parse_evalplus_result(response)
    except ValueError:
        return None, "it does not contain a ```python code block."

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code, f"the code is not valid Python ({e.msg}, line {e.lineno})."

    # Only module-level names count: the tests call the entry point at module scope.
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    defined |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if entry_point not in defined:
        return code, f"it does not define a function named exactly `{entry_point}`."

    if not degenerate:
        return code, None

    # Not in Pinpoint: a body that does nothing parses fine but can never pass a test.
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        ),
        None,
    )
    kind = _degenerate(function) if function is not None else None
    if kind == "empty":
        return code, (
            f"the body of `{entry_point}` does nothing (it only passes, raises or returns "
            "nothing); write the complete implementation."
        )
    if kind == "constant":
        return code, (
            f"`{entry_point}` ignores its arguments and always returns the same literal; write "
            "the complete implementation that computes the result from its arguments."
        )

    return code, None


def _degenerate(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """ "empty" for a placeholder body (only pass / raise / docstrings / a bare `return`),
    "constant" for a function with arguments whose whole body returns one literal,
    None for a real implementation."""
    body = [
        stmt
        for stmt in function.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]
    if all(
        isinstance(stmt, (ast.Pass, ast.Raise))
        or (
            isinstance(stmt, ast.Return)
            and (
                stmt.value is None
                or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
            )
        )
        for stmt in body
    ):
        return "empty"
    has_arguments = bool(
        function.args.posonlyargs
        or function.args.args
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
    )
    if has_arguments and len(body) == 1 and isinstance(body[0], ast.Return):
        try:
            ast.literal_eval(body[0].value)
            return "constant"
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    return None


def _is_placeholder(code: str | None, entry_point: str) -> bool:
    """True when there is no code, or the entry point's body is an empty placeholder."""
    if not code:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return _degenerate(node) == "empty"
    return False


def _alias_entry_point(code: str, entry_point: str) -> str | None:
    """Last resort when repairs fail: alias the only top-level function to the entry point."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    top_level = [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(top_level) != 1:
        return None
    return f"{code.rstrip()}\n\n{entry_point} = {top_level[0]}\n"


def _finalize(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    response: str,
    clarification: str,
    degenerate: bool = True,
) -> str:
    """Validates the response, repairing it a bounded number of times; returns best code.
    degenerate=False: exactly Pinpoint's version (see _check_code)."""
    entry_point = problem["entry_point"]
    best = ""
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        code, issue = _check_code(response, entry_point, degenerate)
        if code is not None and issue is None:
            return code
        if code is not None and not best:
            best = code
        if attempt == MAX_REPAIR_ATTEMPTS:
            break
        response = env.llm(
            _fill(
                REPAIR_TEMPLATE,
                problem=issue,
                prompt=problem["prompt"],
                clarification=clarification,
                response=response,
                entry_point=entry_point,
            )
        )

    return _alias_entry_point(best, entry_point) or best


# ---------------------------------------------------------------------------------------------
# 5. Verification by execution: probe calls, arbitration, choice among implementations
# ---------------------------------------------------------------------------------------------
# verify_by_execution writes probe calls (PROBES_TEMPLATE), runs `candidates` implementations
# on them in the sandbox (VERIFY_SCRIPT), repairs crashes that look like bugs (BUG_EXCEPTIONS),
# arbitrates results that differ from the expected ones (ARBITRATION_TEMPLATE) and keeps the
# implementation with the fewest problems, then the behaviour most implementations share.

MAX_PROBES = 6


MAX_REPR = 4000


MAX_ARBITRATIONS = 2


BUG_EXCEPTIONS = frozenset(
    {
        "TypeError",
        "NameError",
        "AttributeError",
        "UnboundLocalError",
        "IndexError",
        "KeyError",
        "ZeroDivisionError",
        "RecursionError",
        "TimeoutError",
        "ImportError",
        "ModuleNotFoundError",
    }
)


MISSING_ARGUMENT = re.compile(r"missing \d+ required (?:positional|keyword-only) argument")


ITERATOR_REPR = re.compile(
    r"^<(?:generator|map|filter|zip|enumerate|reversed|\w+_iterator|itertools\.\w+) object"
)


PROBES_TEMPLATE = """
Write 4 different calls to the Python function `{entry_point}` exactly as a hidden test would
make them -- three typical ones and one edge case (e.g. an empty or a one-element input) --
based only on the task and the clarification below: pass the arguments in the number, order and
types they imply, as Python literals, and keep the inputs small and simple.
When the clarification makes the exact result of a call certain and it is easy to work out by
hand, add it after ` == ` as a Python literal; otherwise write the call alone.
Write only the calls, one per line:
CALL {entry_point}(...)
CALL {entry_point}(...) == <result>

Task:
<<<
{prompt}
>>>
{clarification}
""".strip()


RESULTS_TEMPLATE = """
Task:
<<<
{prompt}
>>>
{clarification}
For each call to `{entry_point}` below, write the exact result that the task and the
clarification make certain, as a Python literal, or UNSURE when they do not determine it or it
is not easy to work out by hand.
One line per call, in the same order, nothing else:
RESULT <python literal or UNSURE>

{calls}
""".strip()


ARBITRATION_TEMPLATE = """
Task:
<<<
{prompt}
>>>
{clarification}
The function `{entry_point}` was called as `{call}`. Two results are proposed:
A: {first}
B: {second}
Work out step by step which result the task and the clarification require (where they conflict,
the clarification wins). End with exactly one line: CORRECT: A, CORRECT: B or CORRECT: NEITHER.
""".strip()


VERDICT = re.compile(r"CORRECT\W*(A|B|NEITHER)\b", re.I)


VERIFY_SCRIPT = """
import ast
import base64
import copy
import signal


def on_alarm(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, on_alarm)
codes, entry_point, calls = ast.literal_eval(base64.b64decode("{payload}").decode())
for number, code in enumerate(codes):
    namespace = {"__name__": "candidate"}
    try:
        signal.setitimer(signal.ITIMER_REAL, 3)
        exec(compile(code, "candidate", "exec"), namespace)
        signal.setitimer(signal.ITIMER_REAL, 0)
        function = namespace[entry_point]
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        print("CAND", number, "LOAD", type(error).__name__, repr(str(error)[:200]).replace("\\n", " "))
        continue
    for index, args in enumerate(calls):
        try:
            signal.setitimer(signal.ITIMER_REAL, {seconds})
            result = function(*copy.deepcopy(args))
            text = repr(result)
            signal.setitimer(signal.ITIMER_REAL, 0)
            kind = "NONE" if result is None else ("VALUE" if len(text) <= {limit} else "LONG")
            print("CAND", number, "CALL", index, kind, text[:{limit}].replace("\\n", " "))
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            message = (type(error).__name__ + ": " + str(error)[:200]).replace("\\n", " ")
            print("CAND", number, "CALL", index, "EXC", message)
""".strip()


CALL_RESULT = re.compile(r"^CAND (\d+) CALL (\d+) (VALUE|NONE|LONG|EXC) ?(.*)$", re.M)


LOAD_RESULT = re.compile(r"^CAND (\d+) LOAD (\w+) (.*)$", re.M)


CALL_SECONDS = 1.0


LINE_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?(?:>>>\s*)?`*\s*(?:CALL\b\s*:?\s*)?(?:assert\s+)?",
    re.I,
)


ARROW = re.compile(r"\s+(?:->|→|=>|returns|should return)\s+", re.I)


RESULT_TAG = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?`*\s*RESULT\b\s*:?\s*", re.I)


ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _arithmetic(node: Any) -> Any:
    """The value of a small arithmetic expression over numbers ("16 + 256", "2 ** 10")."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _arithmetic(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
        left, right = _arithmetic(node.left), _arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 10**6):
            raise ValueError("too large")
        return ARITHMETIC[type(node.op)](left, right)
    raise ValueError("not arithmetic")


def _literal_node(node: Any) -> tuple[bool, Any]:
    """(True, value) for a literal (or plain arithmetic) that survives the trip into the
    sandbox, else (False, None)."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        try:
            value = _arithmetic(node)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
            return False, None
    try:
        ast.literal_eval(repr(value))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None
    return True, value


def _expression(text: str) -> Any:
    """The parsed expression of a probe or result line, or None."""
    text = LINE_PREFIX.sub("", text, count=1).strip().strip("`").rstrip(";").strip()
    for candidate in (text, ARROW.sub(" == ", text, count=1)):
        try:
            return ast.parse(candidate, mode="eval").body
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
    return None


def parse_probe_calls(response: str, entry_point: str) -> list[tuple[str, tuple, tuple]]:
    """Calls of the entry point with literal arguments, one per line, optionally followed by
    `== result`, as (call text, args, (has_expected, expected))."""
    calls = []
    for line in response.splitlines():
        node = _expression(line)
        expected: tuple = (False, None)
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                expected = _literal_node(node.comparators[0])  # a non-literal is just unchecked
            node = node.left
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == entry_point
            and not node.keywords
        ):
            continue
        found, args = _literal_node(ast.Tuple(elts=node.args, ctx=ast.Load()))
        if not found:
            continue
        calls.append((ast.unparse(node), tuple(args), expected))
    return calls


def _result_value(line: str) -> tuple[bool, Any]:
    text = RESULT_TAG.sub("", line, count=1).strip().strip("`").strip()
    if not text or text.upper().startswith("UNSURE"):
        return False, None
    node = _expression(text)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        node = node.comparators[0]  # the model echoed the call: "f(1) == 2"
    return _literal_node(node) if node is not None else (False, None)


def parse_results(response: str, count: int) -> list[tuple[bool, Any]]:
    """One result per line, in order (the RESULT tag is optional), padded with "unsure"."""
    lines = [
        line
        for line in response.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    tagged = [line for line in lines if RESULT_TAG.match(line)]
    if tagged:
        results = [_result_value(line) for line in tagged]
    else:  # untagged: values and UNSURE only, so that a line of prose does not shift the order
        results = [
            value
            for line in lines
            for value in [_result_value(line)]
            if value[0] or "UNSURE" in line.upper()
        ]
    return (results + [(False, None)] * count)[:count]


def _repr_value(text: str) -> tuple[bool, Any]:
    """The value of a repr printed by the sandbox, when it is a literal."""
    try:
        return _literal_node(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False, None


def _kind(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "sequence"
    if isinstance(value, dict):
        return "mapping"
    return repr(type(value))  # no dunder access: the anti-cheat check rejects it


def same_shape(a: Any, b: Any) -> bool:
    """True when two results are the same kind of value, recursively: numbers (bool, int and
    float alike), strings, None, sequences (list, tuple and set alike, compared item by item
    over the common length) and mappings. Values and lengths are not compared."""
    kind = _kind(a)
    if kind != _kind(b):
        return False
    if kind == "sequence":
        xs = sorted(a, key=repr) if isinstance(a, (set, frozenset)) else a
        ys = sorted(b, key=repr) if isinstance(b, (set, frozenset)) else b
        return all(same_shape(x, y) for x, y in zip(xs, ys, strict=False))
    if kind == "mapping":
        return all(same_shape(a[key], b[key]) for key in a.keys() & b.keys())
    return True


def same_value(a: Any, b: Any) -> bool:
    """Lenient equality: a list equals a tuple with the same items, a set equals the same items
    in any order, floats are equal within 1e-6."""
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        if not all(isinstance(v, (set, frozenset, list, tuple)) for v in (a, b)):
            return False
        a, b = sorted(a, key=repr), sorted(b, key=repr)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        except TypeError:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _result(kind: str, info: str) -> tuple[bool, Any]:
    """(True, value) for a sandbox result that can be compared, else (False, None)."""
    if kind == "NONE":
        return True, None
    if kind == "VALUE":
        return _repr_value(info)
    return False, None


def value_mismatches(report: dict[str, Any], probes: list[tuple[str, tuple, tuple]]) -> list[int]:
    """Calls whose result is of the expected kind but not the expected value."""
    if report["load_error"]:
        return []
    indices = []
    for index, ((kind, info), (_, _, (has_expected, expected))) in enumerate(
        zip(report["results"], probes, strict=True)
    ):
        found, value = _result(kind, info) if has_expected else (False, None)
        if found and same_shape(value, expected) and not same_value(value, expected):
            indices.append(index)
    return indices


def arbitrate(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    clarification: str,
    call: str,
    got: str,
    expected: Any,
) -> bool:
    """True when an independent step-by-step check sides with the expected result."""
    response = env.llm(
        _fill(
            ARBITRATION_TEMPLATE,
            prompt=problem["prompt"],
            clarification=clarification,
            entry_point=problem["entry_point"],
            call=call,
            first=got,
            second=repr(expected),
        )
    )
    verdicts = VERDICT.findall(response or "")
    return bool(verdicts) and verdicts[-1].upper() == "B"


def run_candidates(
    env: ClarificationEnvironment, codes: list[str], entry_point: str, calls: list[tuple]
) -> list[dict[str, Any] | None] | None:
    """One sandbox run of every implementation on every call: a report {load_error, results:
    [(kind, info)]} per implementation (None when its output is incomplete), or None when the
    sandbox gave no output at all. kind is VALUE / NONE / LONG (repr too long) / EXC."""
    payload = repr((codes, entry_point, calls))
    try:
        ast.literal_eval(payload)
        output = env.exec_code(
            _fill(
                VERIFY_SCRIPT,
                payload=base64.b64encode(payload.encode()).decode(),
                limit=str(MAX_REPR),
                seconds=str(CALL_SECONDS),
            )
        )
    except Exception:
        return None
    output = output or ""
    loads = {int(m.group(1)): (m.group(2), m.group(3)) for m in LOAD_RESULT.finditer(output)}
    found: dict[int, dict[int, tuple[str, str]]] = {}
    for m in CALL_RESULT.finditer(output):
        found.setdefault(int(m.group(1)), {})[int(m.group(2))] = (m.group(3), m.group(4))
    reports: list[dict[str, Any] | None] = []
    for number in range(len(codes)):
        results = found.get(number, {})
        if number in loads:
            reports.append({"load_error": loads[number], "results": []})
        elif sorted(results) == list(range(len(calls))):
            reports.append({"load_error": None, "results": [results[i] for i in range(len(calls))]})
        else:
            reports.append(None)
    return reports if any(report is not None for report in reports) else None


def run_calls(
    env: ClarificationEnvironment, code: str, entry_point: str, calls: list[tuple]
) -> dict[str, Any] | None:
    """The sandbox report of one implementation (see run_candidates)."""
    reports = run_candidates(env, [code], entry_point, calls)
    return reports[0] if reports else None


def _same_result(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Whether two implementations behaved the same on one call."""
    (kind_a, info_a), (kind_b, info_b) = a, b
    if kind_a != kind_b:
        return False
    if kind_a == "EXC":
        return info_a.split(":", 1)[0] == info_b.split(":", 1)[0]
    if kind_a == "VALUE":
        (found_a, value_a), (found_b, value_b) = _repr_value(info_a), _repr_value(info_b)
        if found_a and found_b:
            return same_value(value_a, value_b)
    return info_a == info_b


def choose_candidate(
    reports: list[dict[str, Any] | None], probes: list[tuple[str, tuple, tuple]]
) -> int | None:
    """The implementation to keep: fewest problems (crashes, wrong kinds of result, ...), then
    the behaviour most implementations share, then most agreed expected values matched, then
    the earliest (the original). None when no implementation could be run."""
    usable = [
        (number, report)
        for number, report in enumerate(reports)
        if report is not None and not report["load_error"]
    ]
    if not usable:
        return None

    def key(item: tuple[int, dict[str, Any]]) -> tuple:
        number, report = item
        problems = len(call_issues(report, probes))
        votes = sum(
            all(
                _same_result(x, y) for x, y in zip(report["results"], other["results"], strict=True)
            )
            for _, other in usable
        )
        matches = sum(
            1
            for (kind, info), (_, _, (has_expected, expected)) in zip(
                report["results"], probes, strict=True
            )
            if has_expected
            and _result(kind, info)[0]
            and same_value(_result(kind, info)[1], expected)
        )
        return (problems, -votes, -matches, number)

    return min(usable, key=key)[0]


def alternative_implementations(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    clarification: str,
    rules: str,
    count: int,
) -> list[str]:
    """Up to `count` more implementations written independently from the same prompt; broken
    ones and placeholders are dropped (no repair calls are spent on them)."""
    alternatives = []
    for _ in range(max(count, 0)):
        try:
            response = env.llm(
                _fill(
                    CODE_TEMPLATE,
                    prompt=problem["prompt"],
                    clarification=clarification,
                    rules=rules,
                )
            )
        except Exception:
            break  # e.g. the budget is spent
        code, issue = _check_code(response, problem["entry_point"])
        if code and issue is None and not _is_placeholder(code, problem["entry_point"]):
            alternatives.append(code)
    return alternatives


def call_issues(
    report: dict[str, Any],
    probes: list[tuple[str, tuple, tuple]],
    value_checked: frozenset[int] = frozenset(),
) -> list[tuple[str, int, str]]:
    """(kind, call index, message) for each problem; kind is "load", "crash", "iterator",
    "shape" (a kind of result the agreed expectation rules out), "value" (a wrong value, only for
    the calls in `value_checked`, whose expected value was confirmed) or "none" (None for every
    call)."""
    if report["load_error"]:
        kind, message = report["load_error"]
        return [("load", -1, f"the code fails when it is loaded ({kind}: {message})")]
    issues = []
    for index, ((kind, info), (text, _, (has_expected, expected))) in enumerate(
        zip(report["results"], probes, strict=True)
    ):
        if kind == "EXC":
            if info.split(":", 1)[0] in BUG_EXCEPTIONS and not MISSING_ARGUMENT.search(info):
                issues.append(("crash", index, f"`{text}` raised {info}"))
            continue
        if kind == "VALUE" and ITERATOR_REPR.match(info):
            issues.append(
                (
                    "iterator",
                    index,
                    f"`{text}` returns {info.split(' at ')[0]}> instead of a finished value",
                )
            )
            continue
        found, value = _result(kind, info) if has_expected else (False, None)
        if not found:
            continue
        if not same_shape(value, expected):
            issues.append(
                (
                    "shape",
                    index,
                    f"for `{text}` the clarification implies a result like {expected!r}, but "
                    f"the code returns {info}, a different kind of value",
                )
            )
        elif index in value_checked and not same_value(value, expected):
            issues.append(
                (
                    "value",
                    index,
                    f"for `{text}` the clarification implies {expected!r}, but the code "
                    f"returns {info}",
                )
            )
    unchecked = [r for r, p in zip(report["results"], probes, strict=True) if not p[2][0]]
    if len(unchecked) >= 2 and all(kind == "NONE" for kind, _ in unchecked):
        issues.append(("none", -1, f"it returns None for every call, e.g. `{probes[0][0]}`"))
    return issues


def better(new: list[tuple[str, int, str]], old: list[tuple[str, int, str]]) -> bool:
    """A repair counts only if it fails fewer checks, loads, and crashes on no new call."""
    old_crashes = {index for kind, index, _ in old if kind == "crash"}
    return (
        len(new) < len(old)
        and all(kind != "load" for kind, _, _ in new)
        and all(index in old_crashes for kind, index, _ in new if kind == "crash")
    )


def expected_results(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    clarification: str,
    written: list[tuple[str, tuple, tuple]],
) -> list[tuple[str, tuple, tuple]]:
    """Keeps an expected result only where a second, independent derivation from the task and
    the answer gives the same kind of value."""
    second = parse_results(
        env.llm(
            _fill(
                RESULTS_TEMPLATE,
                prompt=problem["prompt"],
                clarification=clarification,
                entry_point=problem["entry_point"],
                calls="\n".join(text for text, _, _ in written),
            )
        ),
        len(written),
    )
    return [
        (text, args, expected)
        if expected[0] and other[0] and same_shape(expected[1], other[1])
        else (text, args, (False, None))
        for (text, args, expected), other in zip(written, second, strict=True)
    ]


def verify_by_execution(
    env: ClarificationEnvironment,
    problem: dict[str, Any],
    code: str,
    clarification: str,
    answered: bool,
    rules: str = "",
    candidates: int = 1,
) -> str:
    """The code, another implementation that behaves better on the probe calls, or a repaired
    version that fails fewer of the same checks. Never raises.

    With `candidates` > 1, that many implementations (the code and independent rewrites from
    the same prompt) run on the same calls in one sandbox run; the one with the fewest problems
    whose behaviour most implementations share is kept. A single sample at temperature 0.7 is
    what makes a task pass in one run and fail in the next; the shared behaviour is steadier."""
    prompt, entry_point = problem["prompt"], problem["entry_point"]
    try:
        stub = stub_annotations(prompt, entry_point)
        arity = len(stub[0]) if stub else None
        probes: list[tuple[str, tuple, tuple]] = [
            (example["call"], example["args"], (False, None))  # crash check only: examples
            for example in extract_examples(prompt, entry_point)  # may be the edited part
            if arity is None or len(example["args"]) == arity
        ][:3]
        response = env.llm(
            _fill(
                PROBES_TEMPLATE,
                prompt=prompt,
                entry_point=entry_point,
                clarification=clarification,
            )
        )
        written = [
            p for p in parse_probe_calls(response, entry_point) if all(p[1] != q[1] for q in probes)
        ][: MAX_PROBES - len(probes)]
        if answered and any(has for _, _, (has, _) in written):
            written = expected_results(env, problem, clarification, written)
        else:  # without a real answer only crashes are checked
            written = [(text, args, (False, None)) for text, args, _ in written]
        probes += written
        if not probes:
            return code
        calls = [args for _, args, _ in probes]
        codes = [code] + alternative_implementations(
            env, problem, clarification, rules, candidates - 1
        )
        reports = run_candidates(env, codes, entry_point, calls)
        chosen = choose_candidate(reports, probes) if reports else None
        if chosen is None:
            return code
        code, report = codes[chosen], reports[chosen]
        # One confirmed wrong value is enough to ask for a repair; try at most two.
        confirmed: frozenset[int] = frozenset()
        for index in value_mismatches(report, probes)[:MAX_ARBITRATIONS]:
            text, _, (_, expected) = probes[index]
            if arbitrate(env, problem, clarification, text, report["results"][index][1], expected):
                confirmed = frozenset({index})
                break
        issues = call_issues(report, probes, confirmed)
        if not issues:
            return code
        problem_text = (
            "running it revealed a problem: "
            + "; ".join(message for _, _, message in issues[:3])
            + ". Fix the function, unless a call or an expected result is itself wrong for the "
            "task and the user's answer; then keep the behaviour that is right."
        )
        response = env.llm(
            _fill(
                REPAIR_TEMPLATE,
                problem=problem_text,
                prompt=prompt,
                clarification=clarification,
                response=f"```python\n{code}\n```",
                entry_point=entry_point,
            )
        )
        repaired = _finalize(env, problem, response, clarification)
        if not repaired or _is_placeholder(repaired, entry_point):
            return code
        second_report = run_calls(env, repaired, entry_point, calls)
        if second_report is not None and better(
            call_issues(second_report, probes, confirmed), issues
        ):
            return repaired
        return code
    except Exception:
        return code


# ---------------------------------------------------------------------------------------------
# 6. Signature fixes: argument order and calls that do not bind
# ---------------------------------------------------------------------------------------------
# collection_first swaps a (scalar, collection) parameter pair, judged from how the body uses
# each parameter (parameter_roles). call_adapter appends CALL_ADAPTER, a wrapper that leaves a
# call that binds unchanged and retries one that does not with generic alternatives.

LENGTH_NAMES = frozenset("n length size len_ num_elements count_elements".split())


CALL_ADAPTER = """

def laaser_adapt_calls(function, length_last):
    import copy
    import inspect

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function
    count = len(signature.parameters)

    def binds(args):
        try:
            signature.bind(*args)
            return True
        except TypeError:
            return False

    def sized(value):
        return isinstance(value, (list, tuple, str, dict, set))

    def alternatives(args):
        lengths = [
            i for i in range(1, len(args))
            if type(args[i]) is int and sized(args[i - 1]) and args[i] == len(args[i - 1])
        ]
        if lengths:
            yield tuple(a for i, a in enumerate(args) if i not in lengths)
        if len(args) > count:
            yield tuple(args[:count])
        if count == 1 and len(args) > 1:
            yield (list(args),)
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == count > 1:
            yield tuple(args[0])
        if length_last and len(args) == count - 1:
            measured = next((a for a in args if sized(a)), None)
            if measured is not None:
                yield (*args, len(measured))

    def adapted(*args, **kwargs):
        if kwargs or binds(args):
            return function(*args, **kwargs)
        for other in alternatives(args):
            if binds(other):
                try:
                    return function(*copy.deepcopy(other))
                except Exception:
                    continue
        return function(*args)

    return adapted


{entry} = laaser_adapt_calls({entry}, {length_last})
"""


def call_adapter(code: str, entry_point: str) -> str:
    """The code with CALL_ADAPTER appended for its entry point; unchanged when the entry point
    is not a plain module-level function (it takes *args, **kwargs or keyword-only or
    positional-only parameters)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point]
    if not definitions:
        return code
    args = definitions[-1].args  # the last one is the one Python binds to the name
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        return code
    length_last = bool(args.args) and args.args[-1].arg.lower() in LENGTH_NAMES
    adapter = _fill(CALL_ADAPTER, entry=entry_point, length_last=str(length_last))
    return code.rstrip() + "\n" + adapter


COLLECTION_CALLS = frozenset(
    "len sorted set list sum max min enumerate zip reversed tuple dict any all map filter".split()
)


ORDER_IN_ANSWER = re.compile(
    r"\b(first|second|last)\s+(argument|parameter|input)|\b(argument|parameter|input)s?\s+order"
    r"|\border\s+of\s+(the\s+)?(argument|parameter|input)s?",
    re.I,
)

ORDER_IN_QUESTION = re.compile(r"\bfirst\b[^.?]{0,80}\bsecond\b", re.I)


def order_by_examples(env: ClarificationEnvironment, code: str, problem: dict[str, Any]) -> str:
    """Swap the (scalar, collection) parameters only if the swapped version satisfies more of
    the stated two-argument examples. Without such examples the code is left untouched."""
    entry_point = problem["entry_point"]
    swapped = collection_first(code, entry_point)
    if swapped == code:
        return code
    examples = [e for e in extract_examples(problem["prompt"], entry_point) if len(e["args"]) == 2]
    if not examples:
        return code
    reports = run_candidates(env, [code, swapped], entry_point, [e["args"] for e in examples])
    if not reports or reports[0] is None or reports[1] is None:
        return code

    def score(report: dict[str, Any]) -> int:
        if report["load_error"]:
            return -1
        total = 0
        for (kind, info), example in zip(report["results"], examples, strict=True):
            found, value = _result(kind, info)
            if found and same_value(value, example["expected"], strict=True):
                total += 1
        return total

    return swapped if score(reports[1]) > score(reports[0]) else code


def parameter_roles(function: ast.FunctionDef) -> str:
    """One letter per parameter: C (used as a collection: iterated, indexed, len(), a method
    on it, ...), S (used only as a scalar: arithmetic, comparisons, range()) or ? (unclear)."""
    names = [a.arg for a in function.args.args]
    collection: set[str] = set()
    scalar: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.iter, ast.Name):
            collection.add(node.iter.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            first = node.args[0] if node.args else None
            if node.func.id in COLLECTION_CALLS and isinstance(first, ast.Name):
                collection.add(first.id)
            if node.func.id == "range":
                scalar.update(a.id for a in node.args if isinstance(a, ast.Name))
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            collection.add(node.value.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            collection.add(node.value.id)
        elif isinstance(node, ast.BinOp):
            scalar.update(x.id for x in (node.left, node.right) if isinstance(x, ast.Name))
        elif isinstance(node, ast.Compare):
            scalar.update(x.id for x in [node.left, *node.comparators] if isinstance(x, ast.Name))
    return "".join("C" if n in collection else ("S" if n in scalar else "?") for n in names)


def collection_first(code: str, entry_point: str) -> str:
    """The code with the entry point's (scalar, collection) parameters swapped in its `def`
    line; the code unchanged when the roles are not exactly that or anything is unusual."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point]
    if len(found) != 1:
        return code
    function = found[0]
    args = function.args
    if len(args.args) != 2 or args.defaults or args.vararg or args.kwonlyargs or args.kwarg:
        return code
    if args.posonlyargs or parameter_roles(function) != "SC":
        return code
    first, second = (ast.unparse(a) for a in args.args)
    lines = code.splitlines(keepends=True)
    pattern = re.compile(
        rf"(def\s+{re.escape(entry_point)}\s*\(\s*)({re.escape(first)})(\s*,\s*)"
        rf"({re.escape(second)})(\s*\))"
    )
    line, count = pattern.subn(r"\1\4\3\2\5", lines[function.lineno - 1], count=1)
    if count != 1:
        return code
    lines[function.lineno - 1] = line
    return "".join(lines)


def shows_signature(problem: dict[str, Any]) -> bool:
    """Whether the task text shows the entry point's `def` line."""
    entry = re.escape(problem.get("entry_point") or "")
    return bool(entry) and bool(re.search(rf"^\s*def\s+{entry}\s*\(", problem["prompt"], re.M))


# ---------------------------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------------------------
# The runner (clarify/utils.py, BatchParallelProcessor) restarts every unfinished task of a batch
# from scratch as soon as one task runs longer than 300 s, up to 100 times. LLM calls have no
# timeout of their own, so one slow call can trigger it. TimedEnvironment gives each task a
# wall-clock budget below that limit: every LLM call waits at most LLM_CALL_TIMEOUT_S, a call
# after the deadline raises LimitsExceededException (which every step already handles), and the
# optional steps (z3, verification) are skipped when little time is left.
TIME_BUDGET_S = 240
LLM_CALL_TIMEOUT_S = 90
Z3_MIN_REMAINING_S = 150
VERIFY_MIN_REMAINING_S = 80


def with_timeout(call: Any, seconds: float) -> Any:
    """call() run in a thread; LimitsExceededException if it does not return within seconds."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except BaseException as error:  # re-raised in the caller's thread
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(max(0.0, seconds))
    if thread.is_alive():
        raise LimitsExceededException("An LLM call exceeded its time budget.")
    if "error" in box:
        raise box["error"]
    return box["value"]


class TimedEnvironment:
    """The task's environment with a wall-clock budget (see above); it only forwards calls."""

    def __init__(self, env: ClarificationEnvironment, budget: float):
        self.env = env
        self.deadline = time.monotonic() + budget

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def llm(self, messages: Any) -> str:
        left = self.remaining()
        if left <= 0:
            raise LimitsExceededException("The task's time budget is spent.")
        return with_timeout(lambda: self.env.llm(messages), min(left, LLM_CALL_TIMEOUT_S))

    def ask_human(self, query: str) -> str:
        return self.env.ask_human(query)

    def can_ask(self) -> bool:
        return self.env.can_ask()

    def exec_code(self, code: str) -> str:
        if self.remaining() <= 0:
            raise LimitsExceededException("The task's time budget is spent.")
        return self.env.exec_code(code)


# ---------------------------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------------------------


class LAASeR_Contract(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # z3 checks the task's statements against its stated examples (needs z3-solver).
        "use_solver": True,
        # "evidence": z3 runs on every task and its proven conflict feeds the contract question,
        # which is always asked. "on_code": the contract question is asked only when the audit
        # asks; when the audit would code, z3 looks for a proven conflict and asks about it.
        "z3_mode": "evidence",
        # Drafts written for the contract question; above 1 they run in the sandbox first.
        "contract_drafts": 1,
        # On tasks without a signature, the question names the argument order.
        "contract_order": True,
        # Audits sampled, when no contract question can be written, to choose the question.
        "question_samples": 1,
        # Verify the final function by execution, comparing this many implementations.
        "verify": True,
        "candidates": 3,
        # Signature fixes on tasks without a signature (step 6).
        "reorder_arguments": True,
        "call_adapter": True,
        # Wall-clock seconds per task (the runner restarts a batch after 300 s).
        "time_budget": TIME_BUDGET_S,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        env = TimedEnvironment(env, float(self.config["time_budget"]))
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        rules = _rules(entry_point)
        if RETURN_FORMAT_STATEMENT.search(prompt):
            rules = f"{rules.rstrip()}\n{ANSWER_OVERRIDES_RULE}"
        candidate, clarification, answered = "", "", False
        try:
            candidate, clarification, answered = self.solve(env, problem, rules)
        except (TooManyQuestionException, LimitsExceededException):
            pass
        except Exception:
            pass  # never lose an already-usable candidate to an unexpected failure
        if _is_placeholder(candidate, entry_point):
            candidate = self.code_without_answer(env, problem, rules) or candidate
        if (
            self.config["verify"]
            and not _is_placeholder(candidate, entry_point)
            and env.remaining() > VERIFY_MIN_REMAINING_S
        ):
            try:
                candidate = verify_by_execution(
                    env,
                    problem,
                    candidate,
                    clarification,
                    answered,
                    rules,
                    int(self.config["candidates"]),
                )
            except Exception:
                pass  # keep the candidate when verification runs out of time
        return self.fix_signature(candidate, problem, clarification)

    def solve(
        self, env: ClarificationEnvironment, problem: dict[str, Any], rules: str
    ) -> tuple[str, str, bool]:
        """(code, the clarification it was written with, whether the user really answered)."""
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        audit_prompt, audit, decision, question = self.pinpoint_audit(
            env, prompt, entry_point, rules
        )
        asks = decision == "ASK" and bool(question)
        contract = None
        if self.config["z3_mode"] == "evidence" or asks:
            contract = self.contract_question(env, problem, audit, rules)
        if contract:
            decision, question = "ASK", contract
        else:
            if asks:
                question = self.choose_question(env, prompt, audit_prompt, question)
            decision, question = self.format_question(env, prompt, entry_point, decision, question)
        if decision == "CODE" or question is None:
            # The audit would code: a conflict proven on the task is asked about instead.
            proven = self.z3_question(env, problem, audit)
            if proven:
                decision, question = "ASK", proven
        if decision == "CODE" or (decision is None and question is None):
            candidate = _check_code(audit, entry_point)[0] or ""
            return _finalize(env, problem, audit, "") or candidate, "", False
        return self.code_with_answer(env, problem, rules, question)

    # Step 1 ------------------------------------------------------------------------------

    def pinpoint_audit(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str, rules: str
    ) -> tuple[str, str, str | None, str | None]:
        """(the audit prompt, the audit, its decision as written, its question): Pinpoint's
        defect audit with the shape reference and, when the task states a return format, the
        contrastive-format rule."""
        checklist = checklist_block(prompt)
        if RETURN_FORMAT_STATEMENT.search(prompt):
            checklist += FORMAT_CONTRAST_RULE
        audit_prompt = _fill(
            AUDIT_CHECKLIST_TEMPLATE,
            prompt=prompt,
            entry_point=entry_point,
            rules=rules,
            checklist=checklist,
        )
        audit = env.llm(audit_prompt)
        return audit_prompt, audit, _parse_decision(audit), atomic_question(_parse_question(audit))

    def choose_question(
        self, env: ClarificationEnvironment, prompt: str, audit_prompt: str, first: str
    ) -> str:
        """Fallback: the first audit's question, or a better one from more sampled audits."""
        questions = [first]
        for _ in range(int(self.config["question_samples"]) - 1):
            try:
                other = env.llm(audit_prompt)
            except Exception:
                break  # e.g. the budget is spent
            question = atomic_question(_parse_question(other))
            if question and _parse_decision(other) != "CODE":
                questions.append(question)
        if RETURN_FORMAT_STATEMENT.search(prompt):
            contrasting = [q for q in questions if contrastive(q)]
            if contrasting:
                return contrasting[0]
        distinct: list[str] = []
        for question in questions:
            if all(_normalized(question) != _normalized(q) for q in distinct):
                distinct.append(question)
        try:
            return distinct[judge_questions(env, prompt, distinct)]
        except Exception:
            return distinct[0]

    def format_question(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        decision: str | None,
        question: str | None,
    ) -> tuple[str | None, str | None]:
        """Fallback: on a task that states a return format, a question contrasting that format
        with the natural type, unless the chosen question already does."""
        if not RETURN_FORMAT_STATEMENT.search(prompt):
            return decision, question
        if decision != "CODE" and question and contrastive(question):
            return decision, question
        written = format_type_question(env, prompt, entry_point)
        return ("ASK", written) if written else (decision, question)

    # Step 2 ------------------------------------------------------------------------------

    def z3_question(
        self, env: ClarificationEnvironment, problem: dict[str, Any], audit: str
    ) -> str | None:
        """The question about a conflict proven on the task (None when there is none)."""
        solver = bool(self.config["use_solver"])
        proven = find_certificate(env, problem, solver=solver)
        if proven is None:
            proven = find_interpretation_conflict(env, problem, audit, None, solver)
        return proven["question"] if proven else None

    # Step 3 ------------------------------------------------------------------------------

    def contract_question(
        self, env: ClarificationEnvironment, problem: dict[str, Any], audit: str, rules: str
    ) -> str | None:
        """The contract question, with the z3 evidence (None when it cannot be written)."""
        evidence = ""
        if self.config["z3_mode"] == "evidence" and env.remaining() > Z3_MIN_REMAINING_S:
            evidence = z3_evidence(env, problem, audit, bool(self.config["use_solver"]))
        return contract_question(
            env,
            problem["prompt"],
            problem["entry_point"],
            audit,
            rules,
            evidence,
            int(self.config["contract_drafts"]),
            bool(self.config["contract_order"]),
        )

    # Step 4 ------------------------------------------------------------------------------

    def code_with_answer(
        self, env: ClarificationEnvironment, problem: dict[str, Any], rules: str, question: str
    ) -> tuple[str, str, bool]:
        """Ask the one question and code with the answer."""
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        answer = None
        if question:
            # The environment reads a reply up to its first backtick: ask without backticks.
            question = question.replace("`", "")
        if question and env.can_ask():
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                answer = None
        answered = bool(question) and answer is not None and usable_answer(answer)
        if answered and answer != answer.rstrip():  # cut at a backtick anyway
            answered = len(answer.strip()) >= MIN_CUT_ANSWER
            answer = answer.rstrip() + " [...]"
        if question and answer is not None and not answered:
            clarification = NO_ANSWER_BLOCK  # a refusal or a dodge must not steer the code
        else:
            clarification = _clarification_block(question, answer)
        response = env.llm(
            _fill(CODE_TEMPLATE, prompt=prompt, clarification=clarification, rules=rules)
        )
        candidate = _check_code(response, entry_point)[0] or ""
        code = _finalize(env, problem, response, clarification) or candidate
        return code, clarification, answered

    def code_without_answer(
        self, env: ClarificationEnvironment, problem: dict[str, Any], rules: str
    ) -> str | None:
        """A complete implementation deciding every unclear point by common sense, or None."""
        try:
            response = env.llm(_fill(FALLBACK_TEMPLATE, prompt=problem["prompt"], rules=rules))
            code = _finalize(env, problem, response, "")
        except Exception:
            return None
        return code if code and not _is_placeholder(code, problem["entry_point"]) else None

    # Step 6 ------------------------------------------------------------------------------

    def fix_signature(self, candidate: str, problem: dict[str, Any], clarification: str) -> str:
        """Argument order and calls that do not bind, on the returned code."""
        entry_point = problem["entry_point"]
        if not candidate:
            return candidate
        if (
            self.config["reorder_arguments"]
            and not shows_signature(problem)
            and not ORDER_IN_ANSWER.search(clarification or "")
        ):
            candidate = collection_first(candidate, entry_point)
        if self.config["call_adapter"]:
            candidate = call_adapter(candidate, entry_point)
        return candidate
