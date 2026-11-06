# SPDX-FileCopyrightText: 2026 XikaiChen <SteveChen19991225@outlook.com>
#
# SPDX-License-Identifier: MIT
# ruff: noqa: B905
# fmt: off
"""Assumption-Driven Clarification.

This submission freezes the ADC V0.3.0 implementation.

Core pipeline:
DRAFT -> MINE -> CONTRACT SYNTHESIS -> ANCHOR PROBES
-> TEMPLATE CONTRACT ASK -> FINAL FRESH GENERATION.

The design adapts AssumptionMiner by making the candidate's implicit behavioral
assumptions explicit, then compressing those assumptions into one grounded contract
clarification for single-function tasks.
No training reference or hidden tests are read at runtime.

Team: SteveChen
Team Members: XikaiChen
Main Contact: SteveChen19991225@outlook.com
"""
from __future__ import annotations

import ast
import builtins
import concurrent.futures
import hashlib
import json
import keyword
import logging
import re
import symtable
from typing import Any, Mapping

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import TooManyQuestionException

__version__ = "0.3.0"
LOG = logging.getLogger(__name__)


DRAFT_PROMPT = """Implement the Python task in INPUT JSON.
Define the function named "{entry_point}" and follow the requirement and its examples.
When the requirement leaves a behavior unspecified, make a reasonable provisional choice.
Return the complete Python implementation.

INPUT JSON:
{input_json}"""


MINE_PROMPT = """You are mining implicit behavioral assumptions from a candidate Python implementation.

Compare the original requirement with the candidate code. Identify up to {max_assumptions}
consequential observable behaviors that the candidate has committed to while the original
requirement leaves that behavior open.

Focus on these contract aspects when they are relevant:
- arguments: calling shape, parameter meaning, input type or structure;
- core_behavior: the main semantic rule computed by the function;
- returns: return type, structure, representation or ordering;
- boundary: behavior on boundary, special or exceptional inputs.

Treat signatures, fixed constants, omitted parameters and fixed output representations as
behavioral assumptions when they commit the function to one observable interpretation.

For each assumption, provide:
aspect: one of "arguments", "core_behavior", "returns", "boundary"
assumption: one concrete, independently revisable behavioral commitment embodied by the candidate
rationale: the gap between what the requirement states and the behavior the candidate commits to

Return fewer assumptions when fewer are supported. The order does not imply priority.

Return a JSON object containing an "assumptions" array. Each item contains only:
aspect, assumption, rationale.

Original requirement:
{prompt}

Entry point:
{entry_point}

Candidate implementation:
{candidate_code}"""


CONTRACT_PROMPT = """Describe the current function contract embodied by the candidate implementation.
Use the original requirement as public context and the mined assumptions to make the candidate's
provisional behavioral commitments explicit.

Return a concise JSON object with exactly these fields:
"arguments": the calling shape, input types/structure, and parameter meaning in a short phrase;
"core_rule": the main rule the candidate computes on ordinary valid inputs in one sentence;
"returns": the exact return type, structure, ordering, or representation in a short phrase.

Original requirement:
{prompt}

Entry point:
{entry_point}

Candidate implementation:
{candidate_code}

Mined assumptions:
{assumptions_json}"""


ANCHOR_PROBE_PROMPT = """Design up to {max_probes} small valid positional calls that make the current
candidate contract concrete for a clarification question.

Rank the calls from strongest to weakest by:
1. how many mined assumptions become observable in the call's behavior;
2. how clearly the call demonstrates the contract's main rule;
3. how simple the input is for a user to understand.

Use inputs allowed by the original requirement. Return a Python code block containing only a
PROBES assignment. PROBES is a list of tuples of positional literal arguments. A one-argument
call uses a one-element tuple. The candidate will be executed to obtain the actual results.

Original requirement:
{prompt}

Entry point:
{entry_point}

Current contract:
{contract_json}

Mined assumptions:
{assumptions_json}

Candidate implementation:
{candidate_code}"""


FINAL_PROMPT = """Implement the Python function using the original requirement and the user's clarification.

The user's answer is authoritative for the behavior it addresses, even if it differs from an earlier interpretation.
For behavior not addressed by the clarification, follow the original requirement and its examples.
Do not invent requirements that are not supported by the requirement or the user's answer.

Original requirement:
{prompt}

Entry point:
{entry_point}

Clarification question:
{question}

User answer:
{answer}

Return the complete Python implementation defining "{entry_point}"."""


def _diagnostic_type_label(value: Any) -> str:
    """Return a best-effort type label without private attribute access."""
    rendered = str(type(value))
    if rendered.startswith("<class '") and rendered.endswith("'>"):
        return rendered[8:-2].rsplit(".", 1)[-1]
    return rendered


def _safe_jsonable(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Best-effort conversion for trace logging; logging must never affect execution."""
    if depth > 20:
        return repr(value)[:1000]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return "<recursive>"
    seen.add(marker)
    try:
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if isinstance(key, str):
                    safe_key = key
                else:
                    try:
                        safe_key = repr(key)
                    except Exception:
                        safe_key = f"<{_diagnostic_type_label(key)}>"
                out[safe_key] = _safe_jsonable(item, depth + 1, seen)
            return out
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_safe_jsonable(item, depth + 1, seen) for item in value]
        try:
            return repr(value)[:2000]
        except Exception:
            return f"<{_diagnostic_type_label(value)}>"
    finally:
        seen.discard(marker)


def _log(stage: str, **data: Any) -> None:
    try:
        payload = _safe_jsonable({"stage": stage, **data})
        LOG.info("ADC_V030 %s", json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Trace output is diagnostic only and must never fail a task.
        try:
            LOG.info("ADC_V030 stage=%s trace_serialization_failed", stage)
        except Exception:
            pass


def _complete(env: Any, text: str, stage: str) -> str:
    reply = env.llm([{"role": "user", "content": text}])
    _log(stage, response=reply)
    return reply if isinstance(reply, str) else ""


def _parallel_complete(env: Any, jobs: list[tuple[str, str]], enabled: bool) -> list[str]:
    if not jobs:
        return []
    if not enabled or len(jobs) == 1:
        return [_complete(env, text, stage) for stage, text in jobs]

    def run(job: tuple[str, str]) -> str:
        stage, text = job
        return _complete(env, text, stage)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(run, jobs))


def _limited_literal(node: ast.AST, depth: int = 0) -> Any:
    """Parse model-produced literals without eval or function calls."""
    if depth > 24:
        raise ValueError("literal too deep")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
    if isinstance(node, ast.Name):
        values = {"null": None, "true": True, "false": False,
                  "None": None, "True": True, "False": False}
        if node.id in values:
            return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_limited_literal(x, depth + 1) for x in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("dict unpacking unsupported")
        return {_limited_literal(k, depth + 1): _limited_literal(v, depth + 1)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _limited_literal(node.operand, depth + 1)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("unsupported unary operand")
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("unsupported literal")


def _read_literal(text: str) -> Any:
    tree = ast.parse(text.strip(), mode="eval")
    return _limited_literal(tree.body)


def _object_texts(text: str) -> list[str]:
    if not isinstance(text, str) or len(text) > 100_000:
        return []
    matches: list[str] = []
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth, quote, escape = 0, None, False
        for i in range(start, len(text)):
            char = text[i]
            if escape:
                escape = False
            elif quote is not None:
                if char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start:i + 1])
                    break
    return matches[:12]


def _json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    try:
        value = json.loads(text.strip())
        if isinstance(value, dict):
            return value
    except (ValueError, TypeError):
        pass
    for candidate in _object_texts(text):
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            try:
                value = _read_literal(candidate)
            except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
                continue
        if isinstance(value, dict):
            return value
    return {}


def _get_code(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    blocks = re.findall(r"^[ \t]*```(?:python3?|py)?[ \t]*\r?\n(.*?)^[ \t]*```",
                        raw, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if blocks:
        return blocks[0].strip() + "\n"
    return raw.strip() + "\n" if raw.strip() else ""


def _normalize_code(code: str, entry: str) -> str:
    """Remove ordinary demos and alias one unambiguous generated function to entry."""
    if not code.strip():
        return ""
    try:
        tree = ast.parse(code)
        scope = symtable.symtable(code, "<adc_v030>", "exec")
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    demo_nodes: list[ast.AST] = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("print", "input")):
            demo_nodes.append(node)
            continue
        if isinstance(node, ast.If) and not node.orelse:
            try:
                condition = ast.unparse(node.test)
            except Exception:
                continue
            if condition in ("__name__ == '__main__'", "'__main__' == __name__"):
                demo_nodes.append(node)

    for node in reversed(demo_nodes):
        del lines[node.lineno - 1:node.end_lineno]
    if demo_nodes:
        code = "".join(lines)
        try:
            tree = ast.parse(code)
            scope = symtable.symtable(code, "<adc_v030>", "exec")
        except SyntaxError:
            return code

    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    bound = entry in scope.get_identifiers() and scope.lookup(entry).is_local()
    if not bound and len(functions) == 1:
        code = code.rstrip() + f"\n\n{entry} = {functions[0]}\n"
        _log("entry_alias", source=functions[0], entry_point=entry)

    # Add only conservative spelling aliases for evaluator compatibility.
    try:
        tree = ast.parse(code)
        identifiers = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        identifiers.add(target.id)
        reserved = set(dir(builtins)) | set(keyword.kwlist)
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", entry).lower()
        snake = re.sub(r"_+", "_", snake)
        variants = sorted({entry.lower(), snake} - {entry} - reserved - identifiers)
        if variants and entry in identifiers:
            code = code.rstrip() + "\n" + "\n".join(f"{alias} = {entry}" for alias in variants) + "\n"
            _log("entry_name_aliases", entry_point=entry, aliases=variants)
    except Exception:
        pass
    return code.rstrip() + "\n"


def _code_ready(code: str, entry: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return entry in names


def _format_retry(env: Any, raw: str, entry: str, stage: str) -> str:
    """One syntax/entry-only retry; it does not change the semantic instruction."""
    retry = env.llm([
        {"role": "assistant", "content": raw or ""},
        {"role": "user", "content":
         f'Return syntactically valid Python defining the function "{entry}". Return code only.'},
    ])
    _log(stage + "_format_retry", response=retry)
    return retry if isinstance(retry, str) else ""


def _code_from_response(env: Any, raw: str, entry: str, stage: str, retry: bool) -> str:
    code = _normalize_code(_get_code(raw), entry)
    if _code_ready(code, entry):
        return code
    if retry:
        code = _normalize_code(_get_code(_format_retry(env, raw, entry, stage)), entry)
        if _code_ready(code, entry):
            return code
    return ""


def _normalize_aspect(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "argument": "arguments", "arguments": "arguments", "input": "arguments",
        "inputs": "arguments", "calling": "arguments", "signature": "arguments",
        "core": "core_behavior", "core_behavior": "core_behavior", "behavior": "core_behavior",
        "behaviour": "core_behavior", "rule": "core_behavior", "semantics": "core_behavior",
        "return": "returns", "returns": "returns", "output": "returns", "outputs": "returns",
        "boundary": "boundary", "boundaries": "boundary", "edge": "boundary",
        "error": "boundary", "errors": "boundary",
    }
    return aliases.get(text, "core_behavior")


def _assumptions(obj: dict[str, Any], limit: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    items = obj.get("assumptions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return output
    for item in items:
        if not isinstance(item, dict):
            continue
        aspect = _normalize_aspect(item.get("aspect"))
        assumption = str(item.get("assumption", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not assumption:
            continue
        key = assumption.casefold()
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha1((aspect + "\0" + assumption).encode("utf-8")).hexdigest()[:8]
        output.append({
            "id": "A_" + digest,
            "aspect": aspect,
            "assumption": assumption,
            "rationale": rationale,
        })
        if len(output) >= limit:
            break
    return output


def _contract(obj: dict[str, Any]) -> dict[str, str]:
    if not isinstance(obj, dict):
        return {}
    result = {
        "arguments": str(obj.get("arguments", "")).strip(),
        "core_rule": str(obj.get("core_rule", "")).strip(),
        "returns": str(obj.get("returns", "")).strip(),
    }
    return result if all(result.values()) else {}


def _contract_format_retry(env: Any, raw: str, stage: str) -> str:
    reply = env.llm([
        {"role": "assistant", "content": raw or ""},
        {"role": "user", "content": (
            'Return the same contract as one JSON object with exactly the string fields '
            '"arguments", "core_rule", and "returns".'
        )},
    ])
    _log(stage + "_format_retry", response=reply)
    return reply if isinstance(reply, str) else ""


def _extract_python_block(raw: str) -> str:
    blocks = re.findall(r"^[ \t]*```(?:python3?|py)?[ \t]*\r?\n(.*?)^[ \t]*```",
                        raw or "", re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return blocks[0].strip() if blocks else (raw or "").strip()


def _parse_probes(raw: str, limit: int) -> list[dict[str, Any]]:
    """Parse PROBES element-by-element so one unsupported expression does not drop the batch."""
    source = _extract_python_block(raw)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    value_node: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "PROBES":
                value_node = node.value
                break
    if not isinstance(value_node, (ast.List, ast.Tuple)):
        return []

    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in value_node.elts:
        try:
            item = _limited_literal(element)
        except (ValueError, TypeError, OverflowError, RecursionError):
            continue
        if not isinstance(item, tuple) or len(item) > 8:
            continue
        representation = repr(item)
        if len(representation) > 2500 or representation in seen:
            continue
        seen.add(representation)
        digest = hashlib.sha1(representation.encode("utf-8")).hexdigest()[:8]
        probes.append({"id": "P_" + digest, "args": item, "args_repr": representation})
        if len(probes) >= limit:
            break
    return probes


def _probe_format_retry(env: Any, raw: str, entry: str, limit: int, stage: str) -> str:
    """One format-only retry when no usable probe can be recovered."""
    reply = env.llm([
        {"role": "assistant", "content": raw or ""},
        {"role": "user", "content": (
            f"Return a Python code block containing only a PROBES assignment for {entry}. "
            f"PROBES is a list of at most {limit} tuples of positional literal arguments. "
            "Use Python literals: strings, numbers, booleans, None, lists, tuples, dicts, and sets."
        )},
    ])
    _log(stage + "_format_retry", response=reply)
    return reply if isinstance(reply, str) else ""


_ANCHOR_HARNESS = r"""
import contextlib, copy, io, json, signal

SOURCE = __SOURCE__
ENTRY = __ENTRY__
PROBES = __PROBES__


def alarm_handler(signum, frame):
    raise TimeoutError("probe timeout")

signal.signal(signal.SIGALRM, alarm_handler)


def load(source):
    ns = {"__name__": "candidate"}
    try:
        signal.alarm(5)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, ns)
        signal.alarm(0)
        return {"ok": True, "fn": ns[ENTRY], "error": ""}
    except BaseException as exc:
        signal.alarm(0)
        return {"ok": False, "fn": None, "error": type(exc).__name__ + ": " + str(exc)[:160]}


def call(fn, args):
    try:
        copied = copy.deepcopy(args)
    except Exception:
        copied = args
    try:
        signal.alarm(2)
        with contextlib.redirect_stdout(io.StringIO()):
            value = fn(*copied)
        signal.alarm(0)
        return {"status": "ok", "type": type(value).__name__, "repr": repr(value)[:320], "error": ""}
    except BaseException as exc:
        signal.alarm(0)
        return {"status": "raises", "type": "", "repr": "",
                "error": type(exc).__name__ + ": " + str(exc)[:160]}

loaded = load(SOURCE)
rows = []
for args in PROBES:
    if not loaded["ok"]:
        outcome = {"status": "load_error", "type": "", "repr": "", "error": loaded["error"]}
    else:
        outcome = call(loaded["fn"], args)
    rows.append({"args_repr": repr(args), "outcome": outcome})

print("@@ADC_ANCHOR@@" + json.dumps({"rows": rows}, ensure_ascii=False))
"""


def _execute_anchor_probes(env: Any, code: str, entry: str,
                           probes: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    if not probes or not code:
        return []
    args = [probe["args"] for probe in probes]
    script = (_ANCHOR_HARNESS
              .replace("__SOURCE__", repr(code))
              .replace("__ENTRY__", repr(entry))
              .replace("__PROBES__", repr(args)))
    try:
        raw = env.exec_code(script) or ""
        match = re.findall(r"@@ADC_ANCHOR@@([^\n]*)", raw)
        parsed = json.loads(match[-1]) if match else {}
        rows = parsed.get("rows", []) if isinstance(parsed, dict) else []
    except Exception as exc:
        _log(stage + "_error", error=_diagnostic_type_label(exc) + ": " + str(exc))
        return []

    output: list[dict[str, Any]] = []
    for probe, row in zip(probes, rows):
        if not isinstance(row, dict):
            continue
        output.append({
            "probe_id": probe["id"],
            "args_repr": probe["args_repr"],
            "outcome": row.get("outcome", {}),
        })
    _log(stage, observations=output)
    return output


def _select_anchor(observations: list[dict[str, Any]], max_repr: int = 160) -> dict[str, Any] | None:
    successful = [row for row in observations if row.get("outcome", {}).get("status") == "ok"]
    for row in successful:
        if len(str(row.get("outcome", {}).get("repr", ""))) <= max_repr:
            return row
    return successful[0] if successful else None


def _clean_question(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^\s*(?:question|q)\s*:\s*", "", text.strip(), flags=re.I)
    return " ".join(text.replace("`", "'").split()).strip()


def _phrase(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").replace("`", "'").split())
    return value[:limit].rstrip(" .;,")


def _call_text(entry: str, args_repr: str) -> str:
    body = args_repr.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
        if body.endswith(","):
            body = body[:-1]
    return f"{entry}({body})"


def _outcome_text(outcome: dict[str, Any]) -> str:
    status = outcome.get("status")
    if status == "ok":
        return "returns " + str(outcome.get("repr", ""))
    if status == "raises":
        return "raises " + str(outcome.get("error", "an exception"))
    if status == "load_error":
        return "cannot be evaluated because the implementation does not load"
    return "has no observed result"


def _contract_question(entry: str, contract: dict[str, str],
                       anchor: dict[str, Any] | None) -> str:
    arguments = _phrase(contract.get("arguments"), 220)
    core_rule = _phrase(contract.get("core_rule"), 300)
    returns = _phrase(contract.get("returns"), 220)
    parts = (
        f"About {entry}: my current understanding is that it takes {arguments}, "
        f"{core_rule}, and returns {returns}."
    )
    if anchor is not None:
        call = _call_text(entry, str(anchor.get("args_repr", "()")))
        outcome = anchor.get("outcome", {})
        if outcome.get("status") == "ok":
            parts += f" For example, {call} returns {outcome.get('repr', '')}."
    parts += " Is this the behavior you intend? If not, please clarify what should behave differently."
    return _clean_question(parts)


# Attribution: the executed-anchor contract confirmation and parts of this delivery
# compatibility layer were informed by the public ContrastiveClarifier submission
# (Team Dinh).

_DELIVERY_CALL_ADAPTER = r"""

def _adc_call_adapter(fn, name):
    import copy, functools, inspect

    try:
        _params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        _params = None

    def _alternatives(args):
        alts = []
        positional = None
        required = None
        if _params is not None and not any(p.kind == p.VAR_POSITIONAL for p in _params):
            positional = [p for p in _params
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if positional is not None:
            limit = len(positional)
            required = sum(p.default is p.empty for p in positional)
            sized_types = (list, tuple, str, dict, set)

            companions = [
                i for i in range(1, len(args))
                if isinstance(args[i], int) and not isinstance(args[i], bool)
                and isinstance(args[i - 1], sized_types)
                and args[i] == len(args[i - 1])
            ]
            if companions and len(args) - len(companions) >= required:
                alts.append(tuple(a for i, a in enumerate(args) if i not in companions))

            if len(args) > limit:
                alts.append(tuple(args[:limit]))

            if required == 1 and len(args) > 1:
                alts.append((list(args),))
                alts.append((tuple(args),))

            if (len(args) == 1 and isinstance(args[0], (list, tuple))
                    and required > 1 and len(args[0]) == required):
                alts.append(tuple(args[0]))

        unique = []
        seen = set()
        for alt in alts:
            key = repr(alt)
            if key != repr(args) and key not in seen:
                seen.add(key)
                unique.append(alt)
        return unique

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        namespace = fn.__globals__
        outer = namespace.get(name)
        namespace[name] = fn
        try:
            try:
                saved = copy.deepcopy(args)
            except Exception:
                saved = args
            try:
                return fn(*args, **kwargs)
            except (TypeError, AttributeError) as first:
                for alt in _alternatives(saved):
                    try:
                        return fn(*copy.deepcopy(alt), **kwargs)
                    except Exception:
                        continue
                raise first
        finally:
            namespace[name] = outer

    return wrapper

{entry} = _adc_call_adapter({entry}, {entry_repr})
"""


def _safe_name_variants(entry: str) -> list[str]:
    reserved = set(dir(builtins)) | set(keyword.kwlist)
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", entry).lower()
    snake = re.sub(r"_+", "_", snake)
    return sorted({entry.lower(), snake} - {entry} - reserved)


def _finish_delivery(code: str, entry: str, use_adapter: bool) -> str:
    """Append delivery-only compatibility; never used during analysis or anchor execution."""
    if not code.strip() or not _code_ready(code, entry):
        return code
    finished = code.rstrip()
    if use_adapter:
        finished += _DELIVERY_CALL_ADAPTER.format(entry=entry, entry_repr=repr(entry))
    aliases = _safe_name_variants(entry)
    if aliases:
        finished += "\n" + "\n".join(f"{name} = {entry}" for name in aliases)
    return finished.rstrip() + "\n"
class AssumptionDrivenClarification(ClarificationAlgorithmBase):
    """Assumption-guided contract clarification for single-function tasks."""

    DEFAULT_CONFIG = {
        "max_assumptions": 3,
        "max_probes": 3,
        "format_retry": True,
        "probe_format_retry": True,
        "contract_format_retry": True,
        "call_adapter": True,
    }

    def run(self, env: Any, problem: dict[str, Any]) -> str:
        prompt = problem["prompt"]
        entry = problem["entry_point"]
        if not isinstance(entry, str) or not entry.isidentifier() or keyword.iskeyword(entry):
            raise ValueError("entry_point is not a Python identifier")

        # 1. DRAFT -- unchanged from V0.2.3.
        public = {"prompt": prompt, "entry_point": entry}
        draft_prompt = DRAFT_PROMPT.format(entry_point=entry,
                                           input_json=json.dumps(public, ensure_ascii=False))
        draft_raw = _complete(env, draft_prompt, "draft_raw")
        draft = _code_from_response(env, draft_raw, entry, "draft", self.config["format_retry"])
        if not draft:
            draft = f"def {entry}(*args, **kwargs):\n    return None\n"
        _log("draft", code=draft)

        # 2. MINE -- expose independently revisable behavioral commitments in C0.
        mine_prompt = MINE_PROMPT.format(
            max_assumptions=int(self.config["max_assumptions"]), prompt=prompt,
            entry_point=entry, candidate_code=draft)
        mine_raw = _complete(env, mine_prompt, "mine_raw")
        assumptions = _assumptions(_json_object(mine_raw), int(self.config["max_assumptions"]))
        _log("assumptions", assumptions=assumptions)

        # 3. CONTRACT SYNTHESIS -- reconstruct the overall contract currently embodied by C0.
        contract_prompt = CONTRACT_PROMPT.format(
            prompt=prompt,
            entry_point=entry,
            candidate_code=draft,
            assumptions_json=json.dumps(assumptions, ensure_ascii=False),
        )
        contract_raw = _complete(env, contract_prompt, "contract_raw")
        contract = _contract(_json_object(contract_raw))
        if not contract and self.config.get("contract_format_retry", True):
            repaired = _contract_format_retry(env, contract_raw, "contract")
            contract = _contract(_json_object(repaired))
        _log("contract", contract=contract)

        # 4. ANCHOR PROBES -- candidate calls ordered by how well they expose the mined assumptions.
        probes: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        anchor: dict[str, Any] | None = None
        if contract:
            anchor_prompt = ANCHOR_PROBE_PROMPT.format(
                max_probes=int(self.config["max_probes"]),
                prompt=prompt,
                entry_point=entry,
                contract_json=json.dumps(contract, ensure_ascii=False),
                assumptions_json=json.dumps(assumptions, ensure_ascii=False),
                candidate_code=draft,
            )
            probe_raw = _complete(env, anchor_prompt, "anchor_probe_raw")
            limit = int(self.config["max_probes"])
            probes = _parse_probes(probe_raw, limit)
            if not probes and self.config.get("probe_format_retry", True):
                repaired_raw = _probe_format_retry(env, probe_raw, entry, limit, "anchor_probe")
                probes = _parse_probes(repaired_raw, limit)
            observations = _execute_anchor_probes(env, draft, entry, probes, "anchor_execution")
            anchor = _select_anchor(observations)
        _log("anchor", probes=probes, observations=observations, selected=anchor)

        # 5. TEMPLATE CONTRACT ASK.
        if contract:
            question = _contract_question(entry, contract, anchor)
        else:
            question = (
                f"About {entry}: please clarify the intended input contract, main behavior, "
                "and return form for this function."
            )
        _log("question", question=question, contract=contract, anchor=anchor)

        try:
            if hasattr(env, "can_ask") and not env.can_ask():
                _log("return", origin="draft", reason="clarification_unavailable")
                return "```python\n" + _finish_delivery(draft, entry, bool(self.config.get("call_adapter", True))).rstrip() + "\n```"
            answer = env.ask_human(question)
        except TooManyQuestionException:
            _log("return", origin="draft", reason="clarification_limit")
            return "```python\n" + _finish_delivery(draft, entry, bool(self.config.get("call_adapter", True))).rstrip() + "\n```"
        if not isinstance(answer, str):
            answer = ""
        _log("answer", answer=answer)
        if not answer.strip():
            _log("return", origin="draft", reason="empty_answer")
            return "```python\n" + _finish_delivery(draft, entry, bool(self.config.get("call_adapter", True))).rstrip() + "\n```"

        # 6. FINAL FRESH GENERATION -- unchanged from V0.2.3.
        final_prompt = FINAL_PROMPT.format(
            prompt=prompt, entry_point=entry, question=question, answer=answer)
        final_raw = _complete(env, final_prompt, "final_raw")
        final_code = _code_from_response(env, final_raw, entry, "final", self.config["format_retry"])
        if not final_code:
            final_code = draft
            _log("final_fallback", reason="invalid_final_code")
        _log("return", origin="final", code=final_code)
        return "```python\n" + _finish_delivery(final_code, entry, bool(self.config.get("call_adapter", True))).rstrip() + "\n```"


def assumption_driven_clarification_v030(
    env: Any, problem: dict[str, Any], config: Mapping[str, Any] | None = None
) -> str:
    return AssumptionDrivenClarification(config).run(env, problem)
