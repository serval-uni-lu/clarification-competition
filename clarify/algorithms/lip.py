# SPDX-FileCopyrightText: 2026 Yang Zhang <zhangy2256@mail2.sysu.edu.cn>
# SPDX-License-Identifier: MIT

# Team lip
# Members: Yang Zhang, Zhen Wang, Wenjie Hu (Sun Yat-sen University)
# Main Contact: Yang Zhang <zhangy2256@mail2.sysu.edu.cn>
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import io
import json
import keyword
import math
import re
import tokenize

from clarify.baselines.base import ClarificationAlgorithmBase as _AlgorithmBase

CONTRACT = """Find one unresolved decision in the specification of {entry} that changes
observable behavior. You receive original requirements, a particular candidate, and
measured outputs from candidates on proposed valid calls. Execution shows what a
candidate does, not what the user intended. Do not turn agreement into correctness.
Prefer a valid call where successful candidates produce different values or types.
Choose a single semantic distinction; a boundary example is useful only if it exposes
that distinction. Never manufacture an alternative output or treat a failed call as
an intended behavior. Keep input-domain uncertainty explicit.

Return a JSON object with these fields:
arguments: a concise description of the declared input interface;
returns: the selected candidate's observed return type and structure;
rule: the particular interpretation at issue, with no unsupported universal claim;
probe: the one-based index of the most informative supplied valid call, or null;
alternative: a concise competing interpretation supported by the supplied candidate
             behavior, or null when no executed competitor supports one;
conflict: null, or an object with a and b containing exact requirement quotations
          that cannot both hold for the same applicable case.
A conflict is not merely an unusual demand, a difficult implementation restriction,
or a mismatch between your preferred convention and the user's words. Describe only
what the evidence supports. Do not bundle unrelated choices into a single question."""


COMPILE = """Extract scoped evidence from the actual answer to a clarification question.
The question provides referents, not additional requirements. Keep the literal answer,
its conditions, its prohibitions, and its unresolved ambiguity. A concrete answer
about one call says nothing universal unless the user states a wider rule. A bare
'yes' to a compound question or alternatives does not select every clause. Resolve
'that example', 'the former', or 'that result' only when there is one unambiguous
referent in the supplied question. Never guess a new expected value from a rejection.

Return one fenced python block containing literal assignments only:
CONFIRMS = True, False, or None; True requires unambiguous acceptance of the particular
question example, False requires its rejection, and None means unresolved;
CALLS = [((positional arguments,), expected_value), ...]; include only exact results
stated in the answer or uniquely resolved from its explicit example reference;
PROPERTIES = []; optionally include at most three boolean expressions over args and
out, but only for explicitly universal answer claims, never an example extrapolation.
Use simple comparisons, indexing, arithmetic, and len/isinstance/all/any/sum/min/max/abs
or built-in type calls. Do not use attributes, comprehensions, imports, lambdas, or powers;
EXAMPLES = a list of CONFIRMED, KEEP, DROP, or CONFLICT strings for requirement examples
in their supplied order; KEEP is the default, DROP requires an explicit applicable
answer correction, and unresolved incompatibility is CONFLICT, not deletion;
CONSTRAINTS = a list of dictionaries, each with:
  text: the constraint with its actual conditions retained;
  quote: an exact contiguous substring of the actual answer supporting it;
  scope: 'example', 'general', or 'unresolved';
  polarity: 'require' or 'forbid';
  reference: 'question_example' only for an unambiguous reference to that example,
             otherwise None.
Include negative constraints and implementation restrictions even when no executable
assertion represents them. Do not convert 'not X' into 'therefore Y'. An exact quote
supports provenance but does not by itself justify a scope or inferred expression.
All assignments are extraction proposals subject to source validation, not authority
to overwrite the original requirement. Use empty lists when nothing is supported."""


FINAL = """Implement the function {entry} from these original requirements:
{prompt}

Actual clarification question:
{question}
Actual user answer:
{answer}

Apply the answer only to the cases and claims it addresses. Resolve an example
reference against the actual question. Preserve every unaffected requirement. Do not
promote a local correction, a rejection, or a vague affirmative into a general rule.
Keep explicit prohibitions and implementation restrictions even when no runtime test
can express them. Where an answer clearly corrects the requirement, use that scoped
correction; where a conflict remains unresolved, do not silently erase either source.
Produce a coherent implementation rather than special-casing the displayed calls.
{obligations}"""


OBLIGATIONS = """
Retained evidence and scoped obligations:
{lines}
These observations constrain the implementation within their stated scopes. Passing
these examples is not sufficient evidence for the rest of the input domain.
"""


REPAIR = """Reconcile the implementation with these observed check failures:
{failures}

Read the original requirement, actual question and answer, and retained scoped evidence
in the preceding context before changing behavior. Keep every confirmed constraint,
including negative, conditional, and non-executable constraints. A failing candidate
does not invalidate source evidence, even if every candidate fails. Fix the governing
operation rather than hard-coding the examples. Do not trade a previously satisfied
binding constraint for more provisional passes. Return one fenced Python implementation."""


REVIEW = """Audit the following implementation of {entry} against the actual sources.
Original requirements:
{prompt}
Actual question:
{question}
Actual answer:
{answer}
Candidate:
```python
{code}
```

Check the entire scoped answer, not only executable examples: negative constraints,
conditional behavior, input/output structure, and implementation restrictions remain
in force. The question's unconfirmed assumptions are not facts. A vague yes or one
accepted example does not establish a universal interpretation. Preserve requirements
outside the answer's scope. Treat generated checks and model summaries as fallible;
do not reject a genuine constraint because candidates fail it.
Return exactly OK only if no source-supported contradiction is found. Otherwise return
CONTRADICTION: followed by the exact source quotation and the applicable scope, then
one fenced Python implementation fixing that contradiction while preserving the other
constraints. If the sources do not settle a choice, do not claim a contradiction based
on your preference or invent new behavior to make a check pass."""


CLEAN = """Preserve this original requirement exactly:
{prompt}

Conflicting examples, unusual output formats, and difficult implementation restrictions
are evidence to clarify, not permission to delete. This stage is not authorized to
rewrite, normalize, weaken, or remove any requirement. Reply exactly UNCHANGED."""


GENERATE = """Write a complete Python implementation for the visible function request.
Use the standard library and keep the requested entry point and calling interface.
The original requirement and actual clarification are the only semantic authorities.
An explicit clarification changes only its supported scope. Preserve unmodified
requirements and prohibitions. Generated probes, candidate outputs, and summaries are
fallible aids, not extra requirements. Do not infer the intended operation from a
function name when the description specifies it. Return one fenced Python code block
containing the implementation, without questions or explanatory prose.
"""


ROBUST = """
Build the computation for the whole supported domain, not only the shown examples.
Do not silently coerce argument shapes, wrap outputs, unpack containers, or invent
exceptions to accommodate a guessed evaluator interface. Do not narrow element types,
character sets, signs, ordering, or emptiness beyond source-supported conditions.
Boundary cases should follow the same operation unless the sources specify otherwise.
Respect constraints on algorithms, libraries, mutation, time, and space even when the
available execution checks cannot observe them. Avoid demonstrations, top-level calls,
interactive input, and printing unless the requested interface explicitly needs them.
"""


PROBE = """Prepare up to {num_probes} small calls for {entry_point} using the supplied
requirements and any actual clarification. Vary inputs to expose competing readings
of the operation, not to repeat one answer example with cosmetic substitutions.
Include ordinary cases and relevant boundaries only when they are in the supported
domain. Do not infer expected outputs or introduce a narrower domain from a candidate.

Return one fenced Python block defining INPUTS as a list of positional-argument tuples
and valid(*args) as a boolean predicate for source-supported preconditions. A unary
list argument must be inside a singleton tuple, not spread across arguments. The
predicate may enforce stated type, range, shape, ordering, and relationship conditions;
it must not reject a call simply because its output is unknown or candidates fail.
Avoid speculative calls when their validity cannot be established. Neither a generated
predicate nor candidate consensus proves domain validity or semantic correctness.
"""


FALLBACK_ASK = """Ask one concise question resolving an important uncertainty in the supplied
requirement. Use an actual executed example only if it is supplied with a valid input
and observed output. Without that evidence, ask directly about the unresolved rule;
do not invent a call, output, or claim that a candidate was executed. Keep alternatives
open to correction. Avoid compound yes/no questions that simultaneously confirm the
input domain, output shape, and general behavior. Return only the question as plain text."""


LITERAL_FAILURES = (ValueError, SyntaxError, TypeError, MemoryError, RecursionError)


NEGATION = re.compile(r"\b(?:not|never|instead\s+of|rather\s+than)\b|n't\b", re.I)


_RELATION = re.compile(
    r"\s*,?\s*(?:(?:it|this|that|the function|the result|the output)\s+)?(?:(?:should|must|would|will)\s+)?(?:==>|==|->|=>|→|returns?|returning|gives?|giving|outputs?|equals?|be|is|:)\s*",
    re.I,
)


def _fences(text, languages=("", "python", "py")):
    if not isinstance(text, str):
        return []
    blocks, current, marker = [], None, None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if current is None:
            match = re.fullmatch(r"(`{3,}|~{3,})([^`~]*)", stripped)
            if match and match[2].strip().lower() in languages:
                marker, current = match[1], []
        elif stripped == marker:
            blocks.append("".join(current).strip())
            current, marker = None, None
        else:
            current.append(line)
    return blocks


def extract_fenced_code(response):
    return next((block for block in _fences(response) if block), None)


def extract_code(response, entry_point, declared=None):
    if (
        not isinstance(response, str)
        or not isinstance(entry_point, str)
        or not entry_point.isidentifier()
        or keyword.iskeyword(entry_point)
    ):
        return None
    for source in [*_fences(response), response.strip()]:
        try:
            tree = ast.parse(source)
        except LITERAL_FAILURES:
            continue
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if entry_point in functions:
            return source
        if declared in functions and entry_point not in {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }:
            return source + "\n\n" + entry_point + " = " + declared + "\n"
    return None


def clean_question(text):
    if not isinstance(text, str):
        return ""
    value = text.strip().strip("`").strip()
    label, colon, content = value.partition(":")
    if colon and label.strip().casefold() in {"question", "q"}:
        value = content
    return " ".join(value.split()).replace("`", "'")


def name_variants(entry_point):
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        return []
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", entry_point)
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", words)
    candidates = {entry_point.lower(), words.lower()}
    return sorted(
        name
        for name in candidates
        if name != entry_point
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not hasattr(builtins, name)
    )


def declared_name(prompt, entry_point):
    if not isinstance(prompt, str):
        return None
    found = {
        m[1]
        for m in re.finditer(r"(?m)^\s*(?:>>>\s*)?(?:async\s+)?def\s+([^\W\d]\w*)\s*\(", prompt)
    }
    return next(iter(found)) if len(found) == 1 and entry_point not in found else None


def _tokens(source):
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            line, column = token.end
            end = offsets[line - 1] + column if line <= len(offsets) else len(source)
            yield token, end
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return


def literal_span(text):
    if not isinstance(text, str) or not text.strip():
        return False, None, 0
    start = len(text) - len(text.lstrip())
    source = text[start : start + 2048]
    best = None
    for token, end in _tokens(source):
        if token.type in (
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            continue
        if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
            break
        if (
            token.type == tokenize.NUMBER
            and token.string.endswith(".")
            and token.string[:-1].isdigit()
        ):
            end -= 1
        try:
            tree = ast.parse(source[:end], mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 256:
                return False, None, 0
            value = ast.literal_eval(tree)
        except LITERAL_FAILURES:
            continue
        best = value, end
    if best is None:
        return False, None, 0
    value, end = best
    remainder = source[end:]
    immediate = remainder[:1]
    suffix = remainder.lstrip()
    if immediate and (immediate.isalnum() or immediate == "_"):
        return False, None, 0
    if suffix.startswith(("+", "-", "*", "/", "%", "^", "&", "|", "<", ">", "=", "(", "[")):
        return False, None, 0
    if immediate == "." and len(remainder) > 1 and (remainder[1].isalnum() or remainder[1] == "_"):
        return False, None, 0
    return True, value, start + end


def leading_literal(text):
    accepted, value, consumed = literal_span(text)
    return accepted, value


def edited_text(reply):
    if not isinstance(reply, str) or reply.lstrip().upper().startswith("UNCHANGED"):
        return None
    return next(iter(_fences(reply, ("", "text"))), None)


def accept_deletion(original, edited, names):
    return None


def balanced_prefix(source):
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for token, _end in _tokens(source):
        if token.type != tokenize.OP:
            continue
        if token.string in pairs:
            if not stack or stack.pop() != pairs[token.string]:
                return False
            if not stack:
                return False
        elif token.string in pairs.values():
            stack.append(token.string)
    return True


def literal_calls(text, name):
    if not isinstance(text, str) or not isinstance(name, str) or not name.isidentifier():
        return []
    calls = []
    for match in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\s*\(", text):
        opening = match.end() - 1
        depth, closing = 0, None
        for token, end in _tokens(text[opening : opening + 4096]):
            if token.type == tokenize.OP:
                depth += int(token.string in "([{") - int(token.string in ")]}")
                if depth == 0:
                    closing = opening + end
                    break
        if closing is None:
            continue
        try:
            node = ast.parse(text[match.start() : closing], mode="eval").body
            if not isinstance(node, ast.Call) or node.keywords or len(node.args) > 64:
                continue
            args = tuple(ast.literal_eval(arg) for arg in node.args)
        except LITERAL_FAILURES:
            continue
        calls.append((args, closing))
    return calls


def doctest_output(prompt, end):
    result = []
    for line in prompt[end:].splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith((">>>", "```", chr(34) * 3, chr(39) * 3)):
            break
        result.append(stripped)
    return "\n".join(result)


def _call_relations(text, names, doctests=False):
    result = []
    for name in dict.fromkeys(names):
        for arguments, end in literal_calls(text, name):
            line_start = text.rfind("\n", 0, end) + 1
            prefix = text[line_start:end]
            name_start = prefix.rfind(name)
            context = prefix[:name_start]
            if NEGATION.search(context) or re.search(
                r"\b(?:if|unless|suppose|supposing)\b", context, re.I
            ):
                continue
            suffix = text[end:]
            connector = _RELATION.match(suffix)
            payload = None
            if (
                doctests
                and prefix.lstrip().startswith(">>>")
                and not suffix.split("\n", 1)[0].strip()
            ):
                payload = doctest_output(text, end)
            elif connector:
                payload = suffix[connector.end() :].split("\n", 1)[0]
            if payload is not None:
                ok, expected = leading_literal(payload)
                if ok:
                    result.append((arguments, expected))
    return result


def parse_examples(prompt, names):
    if not isinstance(prompt, str):
        return []
    selected = []
    for arguments, value in _call_relations(prompt, names, True):
        if not stated_in(selected, arguments, value):
            selected.append((arguments, value))
    return selected[:8]


def assigned_literals(code_block, names):
    if not isinstance(code_block, str) or len(code_block) > 65536:
        return {}
    try:
        tree = ast.parse(code_block)
    except LITERAL_FAILURES:
        return {}
    allowed, result, ambiguous = set(names), {}, set()
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        key = statement.targets[0].id
        if key not in allowed or key in ambiguous:
            continue
        try:
            if sum(1 for _ in ast.walk(statement.value)) > 4096:
                continue
            value = ast.literal_eval(statement.value)
        except LITERAL_FAILURES:
            continue
        if key in result:
            result.pop(key)
            ambiguous.add(key)
        else:
            result[key] = value
    return result


def read_block_values(response, names):
    return assigned_literals(extract_fenced_code(response), names)


def is_demo(node):
    if isinstance(node, ast.If):
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in {"print", "input"}
    )


def sanitize(code):
    if not isinstance(code, str):
        return code
    try:
        tree = ast.parse(code)
    except LITERAL_FAILURES:
        return code
    lines = code.splitlines(keepends=True)
    excluded = set()
    for statement in tree.body:
        if is_demo(statement):
            excluded.update(range(statement.lineno - 1, statement.end_lineno))
    return "".join(line for index, line in enumerate(lines) if index not in excluded)


def call_text(entry, args_repr):
    try:
        arguments = ast.literal_eval(args_repr)
    except LITERAL_FAILURES:
        return str(entry) + "(" + str(args_repr).strip() + ")"
    if not isinstance(arguments, tuple):
        arguments = (arguments,)
    return str(entry) + "(" + ", ".join(repr(argument) for argument in arguments) + ")"


def phrase(text, limit=220):
    normalized = " ".join(("" if text is None else str(text)).split()).replace("`", "'")
    return normalized[: max(0, limit)].rstrip(" .;,")


def quoted_in(prompt, quote):
    if not isinstance(prompt, str) or not isinstance(quote, str):
        return False
    excerpt = " ".join(quote.strip("\"' ").casefold().split())
    return len(excerpt) >= 8 and excerpt in " ".join(prompt.casefold().split())


def answer_relations(answer, names):
    if not isinstance(answer, str):
        return []
    relations = _call_relations(answer, names)
    pattern = re.compile(
        r"\bfor\s+(?:(?:the|a|an)\s+)?(?:input|argument|value|call|n|x)?\s*(?:of\s+|=\s*)?", re.I
    )
    starts = [(match.end(), match.start()) for match in pattern.finditer(answer)]
    starts.extend(
        (match.start(), match.start()) for match in re.finditer(r"(?<![\w])(?=[\[\(\'\"])", answer)
    )
    for start, context_start in starts:
        ok, argument, consumed = literal_span(answer[start : start + 2048])
        if not ok:
            continue
        connector = _RELATION.match(answer, start + consumed)
        if connector is None or NEGATION.search(
            answer[max(0, context_start - 40) : connector.end()]
        ):
            continue
        accepted, expected = leading_literal(answer[connector.end() :].split("\n", 1)[0])
        if accepted and not stated_in(relations, (argument,), expected):
            relations.append(((argument,), expected))
    return relations


def stated_in(relations, args, expected):
    target = (tuple(args), expected)
    return any(exact_value((tuple(arguments), value), target) for arguments, value in relations)


def safe_property(expr):
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 200:
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except LITERAL_FAILURES:
        return None
    functions = {
        "len",
        "isinstance",
        "all",
        "any",
        "sorted",
        "sum",
        "abs",
        "min",
        "max",
        "type",
        "list",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "set",
    }
    permitted = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Compare,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.Call,
        ast.cmpop,
        ast.boolop,
        ast.unaryop,
        ast.operator,
    )
    for node in ast.walk(tree):
        if not isinstance(node, permitted):
            return None
        if isinstance(node, ast.Name) and node.id not in functions | {"args", "out"}:
            return None
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in functions or node.keywords
        ):
            return None
        if isinstance(node, (ast.Pow, ast.MatMult)):
            return None
    return expr.strip()


def exact_value(left, right):
    pending, seen = [(left, right)], set()
    while pending:
        a, b = pending.pop()
        if type(a) is not type(b):
            return False
        pair = (id(a), id(b))
        if pair in seen:
            continue
        seen.add(pair)
        if type(a) in (list, tuple):
            if len(a) != len(b):
                return False
            pending.extend(zip(a, b, strict=False))
        elif type(a) is dict:
            if len(a) != len(b):
                return False
            unmatched = list(b.items())
            for key, value in a.items():
                match = next(
                    (
                        index
                        for index, (other, _) in enumerate(unmatched)
                        if exact_value(key, other)
                    ),
                    None,
                )
                if match is None:
                    return False
                _, other_value = unmatched.pop(match)
                pending.append((value, other_value))
        elif type(a) in (set, frozenset):
            if len(a) != len(b) or any(not any(exact_value(x, y) for y in b) for x in a):
                return False
        elif type(a) in (str, bytes, int, float, complex, bool, type(None)):
            if a != b:
                return False
        else:
            return False
    return True


def explicit_example_confirmation(answer):
    if not isinstance(answer, str) or NEGATION.search(answer):
        return False
    clauses = re.split(r"[.;!?\n]", answer.casefold())
    referent = re.compile(r"\b(?:example|call|output|result)\b")
    approval = re.compile(r"\b(?:correct|right|confirmed|confirm|accept)\b")
    return any(referent.search(clause) and approval.search(clause) for clause in clauses)


def reference_relation(answer, example):
    if example is None or not isinstance(answer, str):
        return []
    text = answer.strip()
    try:
        direct = ast.literal_eval(text.removesuffix("."))
    except LITERAL_FAILURES:
        direct = None
    else:
        return [(example[0], direct)]
    reference = re.search(
        r"\b(?:(?:that|this|your|the)\s+(?:example|call|input)|for that|it\s+(?:should|must))\b",
        text,
        re.I,
    )
    if reference is None:
        return []
    result = re.search(
        r"\b(?:should|must)\s+(?:return|give|be)\s+", text[reference.start() :], re.I
    )
    if result is None:
        return []
    end = reference.start() + result.end()
    if NEGATION.search(text[:end]):
        return []
    ok, value = leading_literal(text[end:])
    return [(example[0], value)] if ok else []


def constraint_ledger(answer, compiled, question, example):
    raw = answer.strip() if isinstance(answer, str) else ""
    records = []
    if not raw:
        return records
    records.append(
        dict(
            id="answer.raw",
            source="answer",
            quote=raw,
            text=raw,
            scope="unresolved",
            polarity="unresolved",
            status="authoritative_in_context",
            executable=False,
            question=question,
        )
    )
    proposals = compiled.get("CONSTRAINTS") if isinstance(compiled, dict) else None
    if not isinstance(proposals, list):
        return records
    vague = (
        re.fullmatch(
            r"(?:(?:yes|ok|correct|sure|right|that is correct|that is right)[, .!\s]*)+", raw, re.I
        )
        is not None
    )
    seen = set()
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        quote = proposal.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote.strip() not in raw:
            continue
        quote = quote.strip()
        scope = proposal.get("scope")
        scope = scope if scope in {"example", "general"} and not vague else "unresolved"
        polarity = proposal.get("polarity")
        polarity = polarity if polarity in {"require", "forbid"} else "unresolved"
        identity = (quote, scope, polarity)
        if identity in seen:
            continue
        seen.add(identity)
        interpretation = proposal.get("text")
        records.append(
            dict(
                id="answer." + str(len(records)),
                source="answer",
                quote=quote,
                text=quote,
                interpretation=interpretation if isinstance(interpretation, str) else quote,
                scope=scope,
                polarity=polarity,
                reference=proposal.get("reference"),
                status="requires_semantic_review",
                executable=False,
            )
        )
    return records


def evidence_context(ledger):
    lines = []
    for record in ledger:
        if not isinstance(record, dict):
            continue
        label = str(record.get("id", "unknown"))
        scope = str(record.get("scope", "unresolved"))
        evidence = record.get("text", record.get("quote", ""))
        lines.append(f"- [{label}; scope={scope}] {evidence}")
    return "\n".join(lines)


def compatible_visible_signature(prompt, entry, arguments):
    if not isinstance(prompt, str) or not isinstance(entry, str) or not entry.isidentifier():
        return True
    signatures = re.finditer(r"\bdef\s+" + re.escape(entry) + r"\s*\(", prompt)
    for match in signatures:
        tail = prompt[match.start() : match.start() + 2048]
        depth = 0
        closing = None
        for token, offset in _tokens(tail):
            if token.type != tokenize.OP:
                continue
            if token.string == "(":
                depth += 1
            elif token.string == ")":
                depth -= 1
                if depth == 0:
                    closing = offset
                    break
        if closing is None:
            continue
        try:
            parsed = ast.parse(tail[:closing] + ":\n    pass").body[0].args
        except LITERAL_FAILURES:
            continue
        maximum = len(parsed.posonlyargs) + len(parsed.args)
        minimum = maximum - len(parsed.defaults)
        if len(arguments) < minimum or (parsed.vararg is None and len(arguments) > maximum):
            return False
        if any(default is None for default in parsed.kw_defaults):
            return False
    return True


def selected_question_result(answer, context, example):
    if example is None or not isinstance(context, dict) or not isinstance(answer, str):
        return []
    words = answer.strip().casefold().rstrip(".!").split()
    words = [word for word in words if word not in {"the", "result"}]
    if len(words) != 1:
        return []
    if words[0] in {"first", "former"}:
        return [example]
    choices = context.get("alternative_outputs")
    if words[0] not in {"second", "latter"} or not isinstance(choices, list) or len(choices) != 1:
        return []
    try:
        value = ast.literal_eval(choices[0])
    except LITERAL_FAILURES:
        return []
    return [(example[0], value)]


ARGUMENT_ADAPTER = r"""
def _compatible_call(function, exported_name):
    import functools
    import inspect
    signature = inspect.signature(function)
    @functools.wraps(function)
    def adapted(*arguments, **keywords):
        signature.bind(*arguments, **keywords)
        return function(*arguments, **keywords)
    return adapted

{entry} = _compatible_call({entry}, {entry_name})
"""


EXECUTION_BODY = r"""
import ast
import copy
import hashlib
import json
import math
import signal
import time

deadline = time.monotonic() + 40
_timer = hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm')
def expire(signum, frame):
    raise TimeoutError('operation budget exhausted')
if _timer:
    try:
        signal.signal(signal.SIGALRM, expire)
    except (ValueError, OSError):
        _timer = False

def bounded(action, seconds):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('execution budget exhausted')
    if _timer:
        signal.alarm(max(1, min(seconds, math.ceil(remaining))))
    try:
        result = action()
        if time.monotonic() > deadline:
            raise TimeoutError('execution budget exhausted')
        return result
    finally:
        if _timer:
            signal.alarm(0)

def same_value(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return len(left) == len(right) and all(any(same_value(k, q) and same_value(v, w) for q, w in right.items()) for k, v in left.items())
    if type(left) in (set, frozenset):
        return len(left) == len(right) and all(any(same_value(a, b) for b in right) for a in left)
    if type(left) not in (type(None), bool, int, str, bytes):
        return False
    return left == right

def invoke(function, arguments, limit):
    return bounded(lambda: function(*copy.deepcopy(arguments)), limit)

def render(value, limit=150):
    try:
        return bounded(lambda: repr(value)[:limit], 1)
    except BaseException:
        return '<unavailable representation>'

def fingerprint(value):
    budget = [2000]
    active = set()
    def encode(item, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 40:
            raise ValueError('value exceeds fingerprint budget')
        kind = type(item)
        if kind is type(None):
            return ['none']
        if kind in (bool, int, str):
            if kind is str and len(item) > 100000:
                raise ValueError('string exceeds fingerprint budget')
            return [kind.__name__, item]
        if kind is float:
            return ['float', item.hex()]
        if kind is bytes:
            if len(item) > 50000:
                raise ValueError('bytes exceed fingerprint budget')
            return ['bytes', item.hex()]
        if kind not in (list, tuple, dict, set, frozenset):
            raise ValueError('unsupported output type')
        if id(item) in active:
            raise ValueError('cyclic output')
        active.add(id(item))
        try:
            if kind is dict:
                children = [[encode(k, depth + 1), encode(v, depth + 1)] for k, v in item.items()]
                children.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
            else:
                children = [encode(x, depth + 1) for x in item]
                if kind in (set, frozenset):
                    children.sort(key=lambda x: json.dumps(x, sort_keys=True))
            return [kind.__name__, children]
        finally:
            active.remove(id(item))
    payload = json.dumps(encode(value), ensure_ascii=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('ascii')).hexdigest()

PROPERTY_FUNCTIONS = {'len': len, 'isinstance': isinstance, 'list': list,
    'tuple': tuple, 'str': str, 'int': int, 'float': float, 'bool': bool,
    'dict': dict, 'set': set, 'all': all, 'any': any, 'sorted': sorted,
    'sum': sum, 'abs': abs, 'min': min, 'max': max, 'type': type}
PROPERTY_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.List,
    ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.BoolOp,
    ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.UAdd, ast.USub, ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn, ast.Call, ast.keyword, ast.IfExp)
def property_code(expression):
    if type(expression) is not str or len(expression) > 2000:
        return None
    try:
        tree = ast.parse(expression, mode='eval')
        nodes = list(ast.walk(tree))
        if len(nodes) > 200:
            return None
        for node in nodes:
            if not isinstance(node, PROPERTY_NODES):
                return None
            if isinstance(node, ast.Name) and node.id not in {'args', 'out', *PROPERTY_FUNCTIONS}:
                return None
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in PROPERTY_FUNCTIONS):
                return None
        return compile(tree, '<property>', 'eval')
    except (SyntaxError, ValueError, RecursionError):
        return None

properties = [code for code in (property_code(x) for x in PROPS) if code is not None]
probe_inputs = []
try:
    namespace = {'__name__': 'probe'}
    bounded(lambda: exec(PROBE_CODE, namespace), 5)
    values = namespace.get('INPUTS', [])
    valid = namespace.get('valid')
    if type(values) in (list, tuple) and callable(valid):
        for value in values[:10]:
            arguments = value if type(value) is tuple else (value,)
            try:
                if bounded(lambda: bool(valid(*copy.deepcopy(arguments))), 2):
                    probe_inputs.append(arguments)
            except BaseException:
                pass
except BaseException:
    pass

observations = []
for source in CANDIDATES:
    row = {'load': False, 'checks': [], 'probes': [], 'props': 0}
    observations.append(row)
    try:
        namespace = {'__name__': 'candidate'}
        bounded(lambda: exec(source, namespace), 5)
        function = namespace[ENTRY]
        if not callable(function):
            raise TypeError('entry is not callable')
    except BaseException:
        row['load'] = True
        continue
    for arguments, expected in CHECKS:
        if time.monotonic() >= deadline:
            row['checks'].append([False, 'skipped'])
            continue
        try:
            output = invoke(function, arguments, 2)
            equal = bounded(lambda: same_value(output, expected), 1)
            row['checks'].append([bool(equal), render(output, 120)])
        except BaseException as error:
            row['checks'].append([False, 'raises ' + type(error).__name__])
    for arguments in probe_inputs:
        if time.monotonic() >= deadline:
            row['probes'].append(['skipped', '', ''])
            continue
        try:
            output = invoke(function, arguments, 2)
        except BaseException as error:
            row['probes'].append(['raises', type(error).__name__, ''])
            continue
        try:
            digest = bounded(lambda: fingerprint(output), 1)
        except BaseException:
            row['probes'].append(['skipped', '', ''])
            continue
        row['probes'].append(['ok', digest, render(output)])
        for code in properties:
            scope = dict(PROPERTY_FUNCTIONS, args=arguments, out=output)
            try:
                if bounded(lambda: bool(eval(code, {'__builtins__': {}}, scope)), 1):
                    row['props'] += 1
            except BaseException:
                pass
print('\n@@HARNESS@@' + json.dumps({'rows': observations, 'inputs': [render(x, 2000) for x in probe_inputs]}))
"""


def finish_program(code, entry, config=None):
    settings = config or {}
    if not isinstance(entry, str) or not entry.isidentifier():
        return code
    if re.search(r"def\s+" + re.escape(entry) + r"\s*\(", code) is None:
        return code
    if settings.get("call_adapter", False):
        code = (
            code.rstrip()
            + "\n"
            + ARGUMENT_ADAPTER.replace("{entry}", entry).replace("{entry_name}", repr(entry))
        )
    aliases = name_variants(entry) if settings.get("name_aliases", False) else []
    if aliases:
        code = (
            code.rstrip()
            + "\n\nfor _exported_alias in "
            + repr(aliases)
            + ":\n    if _exported_alias not in globals():\n        globals()[_exported_alias] = "
            + entry
            + "\n"
        )
    return code


def execution_script(codes, entry, checks, probe_code, props=(), config=None):
    if type(entry) is not str or not entry.isidentifier() or len(entry) > 200:
        raise ValueError("Execution requires a Python entry-point identifier")
    try:
        compile("def " + entry + "(): pass", "<entry-validation>", "exec")
    except SyntaxError as error:
        raise ValueError("Entry point is a reserved Python name") from error
    programs, assertions, properties = list(codes), list(checks), list(props)
    if len(programs) > 32 or len(assertions) > 256 or len(properties) > 64:
        raise ValueError("Execution request exceeds the supported batch limits")
    for source in programs:
        if type(source) is not str or len(source) > 1000000:
            raise ValueError("Each candidate must be bounded Python source text")
    probe_source = "" if probe_code is None else probe_code
    if type(probe_source) is not str or len(probe_source) > 100000:
        raise ValueError("Probe source must be bounded text")
    if any(type(item) is not str or len(item) > 2000 for item in properties):
        raise ValueError("Properties must be bounded expression strings")
    remaining, active = [20000], set()

    def require_literal(value, depth=0):
        remaining[0] -= 1
        if depth > 40 or remaining[0] < 0:
            raise ValueError("Check literals exceed the structural budget")
        kind = type(value)
        if kind in (type(None), bool, int):
            return
        if kind in (str, bytes):
            if len(value) > 100000:
                raise ValueError("Check literal exceeds the text budget")
            return
        if kind is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Checks require finite floating-point literals")
            return
        if kind not in (tuple, list, dict, set, frozenset) or id(value) in active:
            raise ValueError("Checks require acyclic built-in literal values")
        active.add(id(value))
        try:
            children = value.items() if kind is dict else value
            for child in children:
                require_literal(child, depth + 1)
        finally:
            active.remove(id(value))

    normalized = []
    for assertion in assertions:
        if type(assertion) not in (list, tuple) or len(assertion) != 2:
            raise ValueError("Each check must contain arguments and an expected value")
        arguments, expected = assertion
        if type(arguments) not in (list, tuple):
            raise ValueError("Check arguments must be a positional sequence")
        require_literal(arguments)
        require_literal(expected)
        normalized.append((tuple(arguments), expected))
    fields = dict(
        PROBE_CODE=probe_source,
        CANDIDATES=[finish_program(source, entry, config) for source in programs],
        ENTRY=entry,
        CHECKS=normalized,
        PROPS=properties,
    )
    declarations = ["{} = {!r}".format(name, value) for name, value in fields.items()]
    return "\n".join(declarations) + "\n" + EXECUTION_BODY


def _unique_report_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate report field")
        result[key] = value
    return result


def parse_execution_output(output, expected_count):
    if (
        type(output) is not str
        or len(output) > 2000000
        or type(expected_count) is not int
        or expected_count < 0
    ):
        return None
    lines = [
        line[len("@@HARNESS@@") :] for line in output.splitlines() if line.startswith("@@HARNESS@@")
    ]
    try:
        result = json.loads(lines[-1], object_pairs_hook=_unique_report_fields)
    except (IndexError, TypeError, ValueError, RecursionError):
        return None
    if type(result) is not dict or set(result) != {"rows", "inputs"}:
        return None
    rows, inputs = result["rows"], result["inputs"]
    if type(rows) is not list or len(rows) != expected_count:
        return None
    if (
        type(inputs) is not list
        or len(inputs) > 10
        or any(type(x) is not str or len(x) > 2000 for x in inputs)
    ):
        return None
    check_count = None
    for row in rows:
        if type(row) is not dict or set(row) != {"load", "checks", "probes", "props"}:
            return None
        if type(row["load"]) is not bool or type(row["props"]) is not int or row["props"] < 0:
            return None
        if type(row["checks"]) is not list or type(row["probes"]) is not list:
            return None
        if any(
            type(x) is not list
            or len(x) != 2
            or type(x[0]) is not bool
            or type(x[1]) is not str
            or len(x[1]) > 120
            for x in row["checks"]
        ):
            return None
        for item in row["probes"]:
            if type(item) is not list or len(item) != 3 or any(type(x) is not str for x in item):
                return None
            status, digest, shown = item
            if status not in ("ok", "raises", "skipped") or len(shown) > 150:
                return None
            if status == "ok" and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                return None
            if status == "raises" and (not digest.isidentifier() or shown):
                return None
            if status == "skipped" and (digest or shown):
                return None
        if row["load"]:
            if row["checks"] or row["probes"] or row["props"]:
                return None
        else:
            if len(row["probes"]) != len(inputs):
                return None
            if check_count is not None and len(row["checks"]) != check_count:
                return None
            check_count = len(row["checks"])
        if row["props"] and not any(x[0] == "ok" for x in row["probes"]):
            return None
    return result


def execute_candidates(env, codes, entry, checks, probe_code, props=(), config=None):
    if not codes:
        return None
    script = execution_script(codes, entry, checks, probe_code, props, config)
    try:
        output = env.exec_code(script)
    except (TimeoutError, OSError, ValueError):
        return None
    result = parse_execution_output(output, len(codes))
    if result is not None:
        for row in result["rows"]:
            if not row["load"] and len(row["checks"]) != len(checks):
                return None
            if row["props"] > len(props) * sum(x[0] == "ok" for x in row["probes"]):
                return None
    return result


class EvidenceGuidedRepairClarifier(_AlgorithmBase):
    DEFAULT_CONFIG = {
        "drafts": 3,
        "finals": 3,
        "probes": 8,
        "repairs": 2,
        "review": True,
        "call_adapter": False,
        "name_aliases": False,
        "clean_contradictions": False,
    }
    DEFAULT_POLICY = {
        "question": "executed_anchor_and_all_candidate_outputs",
        "evidence": "source_grounded_scoped_ledger",
        "selection": "hard_soft_properties_joint_execution_creation_order",
        "repair": "preserve_satisfied_binding_checks_and_increase_passes",
        "review": "scoped_semantic_review_and_nonregression",
        "model_requests": "sequential",
    }

    def __init__(self, config=None):
        settings = self.DEFAULT_CONFIG.copy()
        if config is not None:
            for name, value in dict(config).items():
                if name not in settings:
                    raise ValueError(f"Unsupported configuration field: {name}")
                settings[name] = value
        for name in ("drafts", "finals", "probes", "repairs"):
            minimum = 1 if name in ("drafts", "finals") else 0
            if type(settings[name]) is not int or settings[name] < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("review", "call_adapter", "name_aliases", "clean_contradictions"):
            if type(settings[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        self.config = settings
        self.policy = self.DEFAULT_POLICY.copy()
        self.artifacts = dict()

    @staticmethod
    def digest(text):
        if type(text) is not str:
            raise TypeError("Artifact digests require text")
        payload = text.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def evidence_records(checks, hard_indices=()):
        binding = {index for index in hard_indices if type(index) is int and index >= 0}
        records = []
        for index, pair in enumerate(checks):
            arguments, expected = pair
            records.append(
                dict(
                    index=index,
                    args=repr(arguments),
                    expected=repr(expected),
                    binding=index in binding,
                )
            )
        return records

    @staticmethod
    def budget_observation(value):
        if type(value) is not float or math.isfinite(value):
            return value
        return dict(type="nonfinite_float", value=repr(value))

    @staticmethod
    def error_label(error):
        type_text = str(type(error))
        qualified_name = type_text.removeprefix("<class '").removesuffix("'>")
        return qualified_name.rsplit(".", 1)[-1]

    def complete(self, env, messages, stage="model"):
        snapshot = copy.deepcopy(messages)
        event = dict(stage=stage, messages=snapshot, response=None, status="pending")
        self.artifacts.setdefault("calls", []).append(event)
        try:
            available = {"prompt_cost": env.prompt_cost, "prompt_budget": env.prompt_budget}
            event["budget"] = {
                name: self.budget_observation(value) for name, value in available.items()
            }
            ledger = self.artifacts.setdefault("public_budget_observations", [])
            for name, value in event["budget"].items():
                ledger.append({"attribute": name, "value": copy.deepcopy(value)})
            exhausted = available["prompt_cost"] >= 0.85 * available["prompt_budget"]
            if exhausted:
                event["response"] = ""
                event["status"] = "budget_skipped"
                return ""
            reply = env.llm(copy.deepcopy(snapshot))
        except Exception as error:
            event.update(status="failed", error_type=self.error_label(error))
            raise
        event["response"] = copy.deepcopy(reply)
        event["status"] = "returned"
        return reply or ""

    def sequential_calls(self, env, requests):
        replies = []
        for request in requests:
            stage, conversation = request
            replies.append(self.complete(env, conversation, stage=stage))
        return replies

    @staticmethod
    def code_messages(content):
        if type(content) is not str:
            raise TypeError("Code-generation requirements must be text")
        roles = (("system", GENERATE + ROBUST), ("user", content))
        return [dict(role=role, content=text) for role, text in roles]

    def program_from_response(self, env, response, messages, entry, declared, stage):
        extracted = extract_code(response, entry, declared)
        attempted_format_recovery = extracted is None and bool(response)
        if attempted_format_recovery:
            conversation = copy.deepcopy(messages)
            conversation.append({"role": "assistant", "content": response})
            conversation.append(
                {
                    "role": "user",
                    "content": "The previous response did not provide a parseable implementation. "
                    "Supply Python source defining "
                    + entry
                    + " with the requested behavior. Include no surrounding explanation.",
                }
            )
            replacement = self.complete(env, conversation, stage=stage + ".format")
            extracted = extract_code(replacement, entry, declared)
        program = sanitize(extracted) if extracted else None
        if program == "":
            program = None
        audit = dict(
            id=stage,
            code=program,
            usable=program is not None,
            sha256=None if program is None else self.digest(program),
            format_recovery_attempted=attempted_format_recovery,
        )
        self.artifacts.setdefault("candidates", []).append(audit)
        return program

    def finish(self, code, entry):
        if type(code) is not str or type(entry) is not str or not entry.isidentifier():
            raise ValueError("Finishing requires source text and a valid entry-point name")
        enabled = {
            option: self.config.get(option, False) for option in ("call_adapter", "name_aliases")
        }
        return finish_program(code, entry, enabled)

    def execute(self, env, codes, entry, checks, probe_code, props=(), stage="execution"):
        programs, assertions, properties = list(codes), list(checks), list(props)
        if not programs:
            return None
        script = execution_script(programs, entry, assertions, probe_code, properties, self.config)
        event = dict(
            stage=stage,
            script=script,
            script_sha256=self.digest(script),
            finished_candidates=[self.finish(program, entry) for program in programs],
            raw=None,
            result=None,
            status="pending",
        )
        self.artifacts.setdefault("executions", []).append(event)
        try:
            event["raw"] = env.exec_code(script)
        except (TimeoutError, OSError, ValueError) as error:
            event.update(status="failed", error_type=self.error_label(error))
            return None
        except Exception as error:
            event.update(status="failed", error_type=self.error_label(error))
            raise
        result = parse_execution_output(event["raw"], len(programs))
        if result is not None:
            for row in result["rows"]:
                valid_checks = row["load"] or len(row["checks"]) == len(assertions)
                successful_probes = sum(probe[0] == "ok" for probe in row["probes"])
                if not valid_checks or row["props"] > len(properties) * successful_probes:
                    result = None
                    break
        event.update(
            result=copy.deepcopy(result),
            status="returned" if result is not None else "invalid_report",
        )
        return result

    @staticmethod
    def passed(row, indices):
        checks = row.get("checks", [])
        admitted = {index for index in indices if type(index) is int and 0 <= index < len(checks)}
        return sum(
            isinstance(checks[index], (list, tuple))
            and len(checks[index]) == 2
            and checks[index][0] is True
            for index in admitted
        )

    @staticmethod
    def output_vector(row, candidate_index):
        probes = row.get("probes", [])
        if (
            row.get("load")
            or not probes
            or any(len(probe) != 3 or probe[0] != "ok" for probe in probes)
        ):
            return ("unobserved", candidate_index)
        return ("observed", tuple(probe[1] for probe in probes))

    def agreement(self, rows, candidate_index):
        signature = self.output_vector(rows[candidate_index], candidate_index)
        if signature[0] != "observed":
            return 0
        return sum(self.output_vector(row, index) == signature for index, row in enumerate(rows))

    def rank(self, rows, hard, soft, stage="selection"):
        if not rows:
            raise ValueError("Cannot select from an empty candidate set")
        binding = {index for index in hard if type(index) is int and index >= 0}
        advisory = {index for index in soft if type(index) is int and index >= 0} - binding
        scores = []
        for index, row in enumerate(rows):
            if row.get("load"):
                score = (-1, -1, -1, -1, -index)
            else:
                properties = row.get("props", 0)
                properties = properties if type(properties) is int and properties >= 0 else 0
                score = (
                    self.passed(row, binding),
                    self.passed(row, advisory),
                    properties,
                    self.agreement(rows, index),
                    -index,
                )
            scores.append(score)
        winner = sorted(range(len(rows)), key=lambda index: scores[index], reverse=True)[0]
        self.artifacts.setdefault("selections", []).append(
            {
                "stage": stage,
                "scores": scores,
                "selected_index": winner,
                "binding_indices": sorted(binding),
                "advisory_indices": sorted(advisory),
            }
        )
        return winner

    @staticmethod
    def probe_table(result, entry):
        if not result or not result.get("inputs"):
            return "(no probe inputs)"
        lines = []
        for index, arguments in enumerate(result["inputs"]):
            columns, outcomes = [], set()
            for row in result.get("rows", []):
                if row.get("load"):
                    columns.append("load failed")
                    continue
                probes = row.get("probes", [])
                if index >= len(probes) or len(probes[index]) != 3:
                    columns.append("not observed")
                    continue
                status, identity, display = probes[index]
                if status == "ok":
                    columns.append(display)
                    outcomes.add((status, identity))
                elif status == "raises":
                    columns.append("raises " + identity)
                    outcomes.add((status, identity))
                else:
                    columns.append("not observed")
            suffix = " (drafts disagree)" if len(outcomes) > 1 else ""
            lines.append(
                str(index + 1)
                + ". "
                + call_text(entry, arguments)
                + " -> "
                + " | ".join(columns)
                + suffix
            )
        return "\n".join(lines)

    def question(self, env, prompt, entry, anchor, result, anchor_row):
        # A witness is useful only when both its arguments and observed value can
        # be represented literally. Probe agreement itself is never evidence.
        snapshots = result or {}
        observations = (
            [] if not anchor_row or anchor_row.get("load") else anchor_row.get("probes", [])
        )
        witnesses = []
        for position, raw_arguments in enumerate(snapshots.get("inputs", [])):
            if position >= len(observations):
                break
            observation = observations[position]
            if observation[0] != "ok":
                continue
            try:
                arguments = ast.literal_eval(raw_arguments)
                arguments = arguments if isinstance(arguments, tuple) else (arguments,)
                value = ast.literal_eval(observation[2])
            except (SyntaxError, ValueError, TypeError):
                continue
            if not compatible_visible_signature(prompt, entry, arguments):
                continue
            distinct = {}
            for candidate in snapshots.get("rows", []):
                probes = candidate.get("probes", [])
                if candidate.get("load") or position >= len(probes):
                    continue
                outcome = probes[position]
                if outcome[0] != "ok" or outcome[1] == observation[1]:
                    continue
                try:
                    ast.literal_eval(outcome[2])
                except (SyntaxError, ValueError, TypeError):
                    continue
                distinct.setdefault(outcome[2], None)
            witnesses.append(
                {
                    "probe_index": position,
                    "call": call_text(entry, raw_arguments),
                    "actual_output": observation[2],
                    "check": (arguments, value),
                    "alternatives": list(distinct),
                }
            )
        chosen = next(
            (w for w in witnesses if w["alternatives"]), witnesses[0] if witnesses else None
        )
        context = {
            "anchor_sha256": self.digest(anchor),
            "draft_and_outputs": "Visible task:\n"
            + prompt
            + "\nCandidate:\n"
            + anchor
            + "\nObserved executions (anchor first):\n"
            + self.probe_table(result, entry),
            "alternative_outputs": [],
            "example": None,
            "domain_validation": "generated_predicate_and_visible_signature_not_a_proof",
        }
        if chosen is None:
            fallback_messages = [
                {"role": "system", "content": FALLBACK_ASK},
                {
                    "role": "user",
                    "content": "Visible task:\n"
                    + prompt
                    + "\nCandidate:\n"
                    + anchor
                    + "\nThere is no admissible executed example. Choose one unresolved behavior to ask about; "
                    "do not claim any unobserved result.",
                },
            ]
            wording = self.complete(env, fallback_messages, "contract.fallback")
            check = None
        else:
            context["example"] = {
                key: chosen[key] for key in ("probe_index", "call", "actual_output")
            }
            context["alternative_outputs"] = chosen["alternatives"][:1]
            check = chosen["check"]
            if context["alternative_outputs"]:
                wording = (
                    f"For the single call {chosen['call']}, I observed {chosen['actual_output']} "
                    f"from one candidate and {context['alternative_outputs'][0]} from another. "
                    "What is the intended return value for this call?"
                )
            else:
                wording = (
                    f"For the single call {chosen['call']}, the candidate returned {chosen['actual_output']}. "
                    "What is the intended return value for this call?"
                )
        self.artifacts["question_context"] = context
        return clean_question(wording), check

    def initial_state(self, env, prompt, entry, declared, examples):
        draft_messages = self.code_messages(
            json.dumps(dict(prompt=prompt, entry_point=entry), ensure_ascii=False)
        )
        probe_instruction = PROBE.format(num_probes=int(self.config["probes"]), entry_point=entry)
        tasks = [("draft." + str(n), draft_messages) for n in range(int(self.config["drafts"]))]
        tasks += [
            (
                "probes.before_answer",
                [dict(role="system", content=probe_instruction), dict(role="user", content=prompt)],
            )
        ]
        responses = dict(
            zip((label for label, _ in tasks), self.sequential_calls(env, tasks), strict=False)
        )
        programs = []
        for label, _ in tasks[:-1]:
            parsed = self.program_from_response(
                env, responses.get(label, ""), draft_messages, entry, declared, label
            )
            if parsed:
                programs.append(parsed)
        probes = extract_fenced_code(responses.get("probes.before_answer", ""))
        batch = (
            self.execute(env, programs, entry, examples, probes, stage="drafts")
            if programs
            else None
        )
        winner = 0
        if batch:
            winner = self.rank(batch["rows"], [], list(range(len(examples))), "anchor")
        chosen = programs[winner] if programs else f"def {entry}(*args, **kwargs):\n    return None"
        # Execution rows are stored anchor-first for question construction; draft
        # indices remain in original generation order for provenance.
        if batch:
            row_order = [winner, *[n for n in range(len(programs)) if n != winner]]
            batch = dict(batch, rows=[batch["rows"][n] for n in row_order])
        self.artifacts["anchor"] = dict(
            code=chosen, sha256=self.digest(chosen), draft_index=winner, fallback_stub=not programs
        )
        return dict(
            drafts=programs,
            anchor=chosen,
            anchor_index=winner,
            execution=batch,
            probe_code=probes,
            probe_system=probe_instruction,
        )

    def compile_checks(self, answer, entry, declared, example_check, examples, compiled):
        question = self.artifacts.get("qa", {}).get("question") or ""
        context = self.artifacts.get("question_context") or {}
        names = list(dict.fromkeys(name for name in (entry, declared) if name))
        references = []
        for name in names:
            for arguments, _offset in literal_calls(question, name):
                if not stated_in(references, arguments, None):
                    references.append((arguments, None))
        anchor = example_check if len(references) < 2 else None
        proposals = answer_relations(answer, names)
        proposals.extend(reference_relation(answer, anchor))
        proposals.extend(selected_question_result(answer, context, anchor))
        if (
            anchor is not None
            and explicit_example_confirmation(answer)
            and not context.get("alternative_outputs")
            and not re.search(r"\b(?:first|second|former|latter)\b", str(answer), re.I)
        ):
            proposals.append(anchor)
        distinct = []
        for arguments, expected in proposals:
            if not stated_in(distinct, arguments, expected):
                distinct.append((arguments, expected))
        conflicts = []
        binding = []
        for arguments, expected in distinct:
            others = [
                (args, value)
                for args, value in distinct
                if stated_in([(args, None)], arguments, None)
                and not stated_in([(args, value)], arguments, expected)
            ]
            if others:
                conflicts.append(
                    {
                        "args": repr(arguments),
                        "expected": repr(expected),
                        "status": "unresolved_answer_conflict",
                    }
                )
            else:
                binding.append((arguments, expected))
        retained = list(binding)
        superseded = []
        for original in examples:
            args, expected = original
            correction = [check for check in binding if stated_in([(check[0], None)], args, None)]
            if correction and not stated_in(correction, args, expected):
                superseded.append({"check": repr(original), "reason": "explicit_answer_same_input"})
            elif not stated_in(retained, args, expected):
                retained.append(original)
        hard = list(range(len(binding)))
        soft = list(range(len(binding), len(retained)))
        records = self.evidence_records(retained, hard)
        ledger = constraint_ledger(answer, compiled, question, example_check)
        for record in records:
            source = "answer" if record["binding"] else "original_prompt"
            record.update(source=source, scope="exact_call", polarity="require", status="retained")
            ledger.append(
                {
                    "id": f"check.{record['index']}",
                    "source": source,
                    "quote": str(answer)
                    if record["binding"]
                    else self.artifacts.get("original_prompt", ""),
                    "text": f"{record['args']} -> {record['expected']}",
                    "scope": "example",
                    "polarity": "require",
                    "status": "binding" if record["binding"] else "source_example",
                    "executable": True,
                }
            )
        self.artifacts.update(
            constraint_ledger=ledger,
            checks=records,
            properties=[],
            superseded_examples=superseded,
            unresolved_answer_conflicts=conflicts,
            unverified_compiler_suggestions=copy.deepcopy(compiled),
        )
        return retained, hard, soft, []

    def final_state(
        self, env, prompt, entry, declared, examples, question, answer, example_check, initial
    ):
        visible_examples = [
            f"{call_text(entry, repr(tuple(args)))} -> {value!r}" for args, value in examples
        ]
        compilation_input = "\n".join(
            [
                "Original requirement:",
                prompt,
                "Asked:",
                question,
                "Actual answer:",
                answer,
                "Source examples:",
                *visible_examples,
            ]
        )
        response = self.complete(
            env,
            [dict(role="system", content=COMPILE), dict(role="user", content=compilation_input)],
            "answer.compile",
        )
        suggestions = read_block_values(
            response, {"CONFIRMS", "CALLS", "PROPERTIES", "EXAMPLES", "CONSTRAINTS"}
        )
        checks, binding, provisional, properties = self.compile_checks(
            answer, entry, declared, example_check, examples, suggestions
        )
        self.artifacts["compiled_evidence"] = {"response": response}
        ledger = evidence_context(self.artifacts["constraint_ledger"])
        exact = [
            f"{call_text(entry, repr(tuple(checks[n][0])))} == {checks[n][1]!r}" for n in binding
        ]
        evidence_text = "\nScoped evidence, read with the actual question and answer:\n" + ledger
        if exact:
            evidence_text += "\nRequired exact calls:\n" + "\n".join(exact)
        generation = self.code_messages(
            FINAL.format(
                prompt=prompt,
                entry=entry,
                question=question,
                answer=answer,
                obligations=evidence_text,
            )
        )
        labels = ["final." + str(n) for n in range(int(self.config["finals"]))]
        tasks = [(label, generation) for label in labels]
        tasks += [
            (
                "probes.after_answer",
                [
                    dict(role="system", content=initial["probe_system"]),
                    dict(
                        role="user",
                        content="\n".join([prompt, "Asked:", question, "Actual answer:", answer]),
                    ),
                ],
            )
        ]
        replies = dict(
            zip((label for label, _ in tasks), self.sequential_calls(env, tasks), strict=False)
        )
        candidates = []
        for label in labels:
            candidate = self.program_from_response(
                env, replies.get(label, ""), generation, entry, declared, label
            )
            if candidate:
                candidates.append(candidate)
        if not candidates:
            candidates.append(initial["anchor"])
        probes = (
            extract_fenced_code(replies.get("probes.after_answer", "")) or initial["probe_code"]
        )
        execution = self.execute(env, candidates, entry, checks, probes, properties, "finals")
        satisfies_binding = execution and any(
            not row["load"] and self.passed(row, binding) == len(binding)
            for row in execution["rows"]
        )
        if binding and execution and not satisfies_binding:
            rescue_context = self.code_messages(
                FINAL.format(
                    prompt=prompt,
                    entry=entry,
                    question=question,
                    answer=answer,
                    obligations=OBLIGATIONS.format(lines="\n".join(exact) + "\n" + ledger),
                )
            )
            rescue_reply = self.complete(env, rescue_context, "rescue")
            replacement = self.program_from_response(
                env, rescue_reply, rescue_context, entry, declared, "rescue"
            )
            if replacement:
                expanded = [*candidates, replacement]
                retrial = self.execute(env, expanded, entry, checks, probes, properties, "rescue")
                if retrial:
                    candidates, execution = expanded, retrial
        self.artifacts.update(
            dropped_provisional_indices=[], active_checks=self.evidence_records(checks, binding)
        )
        winner = self.rank(execution["rows"], binding, provisional, "final") if execution else 0
        return dict(
            code=candidates[winner],
            status=execution["rows"][winner]["checks"] if execution else None,
            checks=checks,
            hard=binding,
            messages=generation,
        )

    @staticmethod
    def keeps_binding(previous, current, hard):
        required = {
            index
            for index in hard
            if type(index) is int
            and 0 <= index < len(previous)
            and len(previous[index]) == 2
            and previous[index][0] is True
        }
        survived = {
            index
            for index, item in enumerate(current)
            if isinstance(item, (tuple, list)) and len(item) == 2 and item[0] is True
        }
        return required <= survived

    def correct(self, env, state, prompt, entry, declared, question, answer):
        selected = state["code"]
        results = state["status"]
        checks, binding = state["checks"], state["hard"]

        def evaluate(candidate, stage, require_progress):
            execution = self.execute(env, [candidate], entry, checks, None, stage=stage)
            if not execution or execution["rows"][0]["load"]:
                return None, {"reason": "execution_unavailable_or_load_failed"}
            observed = execution["rows"][0]["checks"]
            before = sum(bool(item[0]) for item in (results or []))
            after = sum(bool(item[0]) for item in observed)
            preserved = len(observed) == len(checks) and self.keeps_binding(
                results or [], observed, binding
            )
            details = dict(
                candidate_sha256=self.digest(candidate),
                preserves_binding=preserved,
                before_passes=before,
                after_passes=after,
            )
            admissible = preserved and (after > before if require_progress else after >= before)
            if not admissible:
                details["reason"] = "binding_loss_or_insufficient_passes"
                return None, details
            if not self.semantic_guard(
                env, candidate, prompt, entry, question, answer, stage + ".guard"
            ):
                details["reason"] = "semantic_veto"
                return None, details
            return observed, details

        rounds = int(self.config["repairs"]) if checks and results else 0
        for number in range(rounds):
            failures = [n for n, item in enumerate(results) if n < len(checks) and not item[0]]
            if not failures:
                break
            diagnostic = []
            for n in failures[:4]:
                arguments, expected = checks[n]
                diagnostic.append(
                    f"- {call_text(entry, repr(tuple(arguments)))}: required {expected!r}; observed {results[n][1]}"
                )
            messages = [
                *state["messages"],
                dict(role="assistant", content="```python\n" + selected + "\n```"),
                dict(role="user", content=REPAIR.format(failures="\n".join(diagnostic))),
            ]
            stage = "repair." + str(number)
            audit = dict(
                round=number,
                failed_indices=failures,
                before_sha256=self.digest(selected),
                accepted=False,
            )
            self.artifacts.setdefault("repairs", []).append(audit)
            candidate = self.program_from_response(
                env, self.complete(env, messages, stage), messages, entry, declared, stage
            )
            if not candidate:
                audit["reason"] = "unusable_program"
                break
            observed, details = evaluate(candidate, stage, True)
            audit.update(details)
            if observed is None:
                break
            selected, results = candidate, observed
            audit.update(accepted=True, reason="binding_preserved_and_pass_count_increased")
        if not self.config["review"]:
            return selected
        review_input = REVIEW.format(
            prompt=prompt, question=question, answer=answer, entry=entry, code=selected
        )
        review_input += "\nScoped evidence:\n" + evidence_context(
            self.artifacts.get("constraint_ledger", [])
        )
        response = self.complete(env, [dict(role="user", content=review_input)], "review")
        clear = response.strip().upper() == "OK"
        audit = dict(
            response=response,
            before_sha256=self.digest(selected),
            accepted=False,
            status="no_contradiction_found" if clear else "unresolved",
            kind="model_review_not_proof",
        )
        self.artifacts["review"] = audit
        if clear or not response:
            return selected
        proposed = extract_code(response, entry, declared)
        if not proposed:
            return selected
        proposed = sanitize(proposed)
        observed, details = evaluate(proposed, "review", False)
        audit.update(details)
        if observed is not None:
            selected = proposed
            audit.update(accepted=True, status="revised_and_reviewed")
        return selected

    def semantic_guard(self, env, code, prompt, entry, question, answer, stage):
        sections = [
            REVIEW.format(prompt=prompt, question=question, answer=answer, entry=entry, code=code),
            "Scoped evidence:",
            evidence_context(self.artifacts.get("constraint_ledger", [])),
        ]
        verdict = self.complete(env, [dict(role="user", content="\n".join(sections))], stage)
        audit = dict(
            stage=stage,
            response=verdict,
            code_sha256=self.digest(code),
            kind="model_review_not_proof",
            accepted=verdict.strip().casefold() == "ok",
        )
        self.artifacts.setdefault("semantic_guards", []).append(audit)
        return audit["accepted"]

    def deliver(self, code, entry, branch):
        output = self.finish(code, entry)
        origin = self.artifacts.get("anchor", {})
        parsed_candidate = any(
            record.get("usable") and record.get("code") == code
            for record in self.artifacts.get("candidates", [])
        )
        selection = dict(
            branch=branch,
            code=code,
            finished_code=output,
            fallback_stub=bool(
                origin.get("fallback_stub") and origin.get("code") == code and not parsed_candidate
            ),
        )
        for source, target in (("code", "sha256"), ("finished_code", "finished_sha256")):
            selection[target] = self.digest(selection[source])
        self.artifacts["selected"] = selection
        return "\n".join(("```python", output, "```"))

    def run(self, env, problem):
        self.artifacts = {
            name: []
            for name in (
                "calls",
                "executions",
                "candidates",
                "selections",
                "repairs",
                "public_budget_observations",
                "can_ask_observations",
            )
        }
        self.artifacts.update(
            config=dict(self.config),
            policy=dict(self.policy),
            review=None,
            actual_qa_count=0,
            qa=dict(
                actual_invocations=0,
                completed_interactions=0,
                question=None,
                answer=None,
                answer_truthy=False,
            ),
        )
        prompt = problem["prompt"]
        entry = problem["entry_point"]
        self.artifacts.update(
            visible_problem=dict(prompt=prompt, entry_point=entry),
            original_prompt=prompt,
            working_prompt=prompt,
            prompt_preprocessing=dict(
                mode="identity", requested_cleaning=self.config["clean_contradictions"]
            ),
        )
        declared = declared_name(prompt, entry)
        names = [entry] if not declared else [entry, declared]
        examples = parse_examples(prompt, names)
        self.artifacts["stated_examples"] = self.evidence_records(examples)
        initial = self.initial_state(env, prompt, entry, declared, examples)
        permission = env.can_ask()
        self.artifacts["can_ask_observations"].append(permission)
        check = None
        interaction = self.artifacts["qa"]
        if permission:
            execution = initial["execution"]
            anchor_row = execution["rows"][0] if execution else None
            wording, check = self.question(
                env, prompt, entry, initial["anchor"], execution, anchor_row
            )
            if wording:
                interaction.update(question=wording, actual_invocations=1)
                returned_answer = env.ask_human(wording)
                interaction.update(
                    answer=returned_answer,
                    answer_truthy=bool(returned_answer),
                    completed_interactions=1,
                )
                self.artifacts["actual_qa_count"] = 1
        if interaction["answer_truthy"]:
            # Every substantive reply is retained verbatim and compiled before
            # fresh generation; mere assent cannot bypass the evidence compiler.
            state = self.final_state(
                env,
                prompt,
                entry,
                declared,
                examples,
                interaction["question"],
                interaction["answer"],
                check,
                initial,
            )
            chosen = self.correct(
                env, state, prompt, entry, declared, interaction["question"], interaction["answer"]
            )
            branch = "post_answer"
        else:
            chosen = initial["anchor"]
            if initial["execution"] and len(initial["drafts"]) > 1:
                order = [
                    initial["anchor_index"],
                    *[n for n in range(len(initial["drafts"])) if n != initial["anchor_index"]],
                ]
                position = self.rank(
                    initial["execution"]["rows"], list(range(len(examples))), [], "without_answer"
                )
                chosen = initial["drafts"][order[position]]
            branch = "without_truthy_answer"
        return self.deliver(chosen, entry, branch)
