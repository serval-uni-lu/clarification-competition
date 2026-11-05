# SPDX-FileCopyrightText: 2026 Yongcheng Huang <Y.Huang-12@tudelft.nl>
#
# SPDX-License-Identifier: MIT

"""ContractFirstClarifier: clarify omissions while preserving explicit rules.

A specification analyst identifies the most consequential missing fact before
any code is written, treating stated constraints as authoritative even when
unconventional. A fresh solver preserves those constraints while incorporating
the actual clarification. Optional self-generated checks run through the SDK
sandbox, with at most one repair. This is an unmeasured experimental candidate.

Team: D4vidHuang
Team Members: Yongcheng Huang
Main Contact: Y.Huang-12@tudelft.nl
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment

_SUCCESS = "QUESTION_FIRST_CHECKS_OK_4A16F08D"


def _main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]
    return any(
        isinstance(name, ast.Name)
        and name.id == "__name__"
        and isinstance(value, ast.Constant)
        and value.value == "__main__"
        for name, value in ((left, right), (right, left))
    )


def _implementation_body(nodes: list[ast.stmt]) -> list[ast.stmt]:
    body = []
    for node in nodes:
        if isinstance(node, ast.Assert):
            continue
        if _main_guard(node):
            body.extend(_implementation_body(node.orelse))
        else:
            body.append(node)
    return body


def sanitize_implementation(code_or_fenced: str, entry_point: str) -> str:
    """Return fenced implementation without module assertions/main demos.

    This deterministic, non-executing helper accepts raw or fenced Python, never
    an LLM or environment. Invalid source or a missing entry point returns "".
    Only module-scope statements are filtered. A conventional main guard's else
    branch is retained as library initialization; function/class bodies and all
    other statements remain intact. Formatting/comments may change via unparse.
    """
    source = _code(code_or_fenced, entry_point)
    if not source:
        return ""
    try:
        tree = ast.parse(source.strip())
        compile(tree, "<candidate>", "exec")
        tree.body = _implementation_body(tree.body)
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body
        ):
            return ""
        cleaned = ast.unparse(tree)
        compile(cleaned, "<implementation>", "exec")
        return "```python\n" + cleaned + "\n```"
    except (SyntaxError, TypeError, ValueError, RecursionError):
        return ""


def _object(text: str | None) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    sources = re.findall(r"```json\s*\n(.*?)```", text, re.S) + [text]
    for source in sources:
        for match in list(re.finditer(r"\{", source))[:32]:
            try:
                value, _ = decoder.raw_decode(source[match.start() :])
                if isinstance(value, dict):
                    return value
            except (ValueError, RecursionError):
                continue
    return {}


def _code(text: str | None, entry_point: str) -> str:
    if not isinstance(text, str):
        return ""
    candidate = _object(text).get("code", "")
    sources = ([candidate] if isinstance(candidate, str) else []) + [text]
    for source in sources:
        for code in re.findall(r"```(?:python|py)?\s*\n(.*?)```", source, re.S) + [source]:
            try:
                tree = ast.parse(code.strip())
                compile(tree, "<candidate>", "exec")  # Validate only; never execute locally.
            except (SyntaxError, TypeError, ValueError, RecursionError):
                continue
            if any(
                isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body
            ):
                return code.strip()
    return ""


def _question(decision: dict[str, Any], threshold: float, max_chars: int) -> str:
    if not (
        decision.get("ask") is True
        and decision.get("atomic") is True
        and decision.get("answer_in_prompt") is False
    ):
        return ""
    gain = decision.get("expected_gain")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not threshold <= gain <= 1:
        return ""
    question, witness = decision.get("question"), decision.get("witness")
    if not isinstance(witness, str) or not witness.strip() or not isinstance(question, str):
        return ""
    question = question.strip()
    if (
        not question
        or len(question) > max_chars
        or "\n" in question
        or question.count("?") != 1
        or not question.endswith("?")
    ):
        return ""
    return question


class ContractFirstClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "max_llm_calls": 3,  # Specification analysis, fresh synthesis, optional repair.
        "max_questions": 1,  # Zero disables the analysis/question stage; one enables it.
        "min_expected_gain": 0.04,  # Estimated absolute correctness gain needed to ask.
        "max_question_chars": 360,  # One concise question about one fact.
        "sandbox_checks": True,  # Execute justified checks only through env.exec_code.
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, str]) -> str:
        task = {"prompt": problem["prompt"], "entry_point": problem["entry_point"]}
        entry_point = task["entry_point"]
        if (
            not isinstance(entry_point, str)
            or not entry_point.isidentifier()
            or keyword.iskeyword(entry_point)
        ):
            raise ValueError("entry_point must be a Python function name")
        max_calls, max_questions = self.config["max_llm_calls"], self.config["max_questions"]
        threshold, max_chars = self.config["min_expected_gain"], self.config["max_question_chars"]
        if type(max_calls) is not int or not 1 <= max_calls <= 3:
            raise ValueError("max_llm_calls must be an integer from 1 to 3")
        if type(max_questions) is not int or max_questions not in (0, 1):
            raise ValueError("max_questions must be 0 or 1")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("min_expected_gain must be between 0 and 1")
        if type(max_chars) is not int or max_chars < 32:
            raise ValueError("max_question_chars must be an integer of at least 32")
        if type(self.config["sandbox_checks"]) is not bool:
            raise ValueError("sandbox_checks must be a boolean")
        calls, stopped = 0, False

        def available() -> bool:
            nonlocal stopped
            if stopped or calls >= max_calls:
                return False
            try:
                stopped = env.prompt_cost >= env.prompt_budget
            except Exception:
                stopped = True
            return not stopped

        def call(instruction: str, payload: dict[str, Any]) -> str | None:
            nonlocal calls, stopped
            if not available():
                return None
            calls += 1
            try:
                return env.llm(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ]
                )
            except Exception:
                stopped = True
                return None

        clarification = None
        can_ask = False
        if max_questions and max_calls >= 2 and available():
            try:
                can_ask = env.can_ask()
            except Exception:
                pass
        if can_ask:
            analysis = call(
                "You are a specification analyst. Do not write an implementation yet. "
                "Find the single missing semantic fact most likely to change the correct "
                "function's observable behavior. Read all signatures and examples first. "
                "Explicit stated rules are authoritative even when unconventional. Never "
                "replace stated numeric bounds, format rules, field widths or definitions "
                "with unstated domain conventions. Ask only about a genuine omission or "
                "internal contradiction; an unconventional rule is not a contradiction. "
                "Prioritize missing core behavior over speculative invalid-input validation. "
                "Mentally compare plausible interpretations on a small input you invent. "
                "Distinguish an omitted requirement from a coding challenge, a mathematical "
                "definition, or something already entailed by the prompt. A conventional "
                "guess is not proof of the intended behavior. Prioritize consequential "
                "omissions that a precise user answer can resolve. Ask exactly ONE fact, "
                "admitting one objective answer. Alternative values of the same fact are "
                "allowed, but do not bundle independent decisions. Never request test cases, "
                "reference code, an algorithm, all missing requirements, or a correctness "
                "guarantee. Do not rely on remembering a benchmark's implementation. "
                "Estimate the absolute gain in correctness probability from obtaining this "
                "one answer: 0.10 means ten percentage points, not merely confidence that "
                "ambiguity exists. Asking has a modest cost, so abstain for cosmetic "
                "details or when the description determines the relevant behavior. "
                f"Keep the question below {max_chars} characters. Return only JSON: "
                '{"ask":true or false,"question":"one concise question ending in ?",'
                '"atomic":true or false,"answer_in_prompt":true or false,'
                '"expected_gain":0.0,"witness":"invented input with differing outputs '
                'under two plausible interpretations","reason":"why this fact matters"}.',
                task,
            )
            question = _question(_object(analysis), threshold, max_chars)
            if question and available():
                try:
                    if env.can_ask():
                        answer = env.ask_human(question)
                        if isinstance(answer, str) and answer.strip():
                            clarification = {"question": question, "answer": answer}
                except Exception:
                    pass

        # Deliberately start a fresh context: no speculative analysis, alternatives,
        # draft code, witness, gain estimate, or analyst reasoning reaches the solver.
        solution = call(
            "Implement the requested Python function as a self-contained module. Use the "
            "original task and any actual clarification answer as the specification. "
            "Before coding, check every explicit constraint, including numeric bounds, "
            "formatting, field widths, signature and return shape; preserve them all even "
            "when they differ from domain conventions. The actual clarification resolves "
            "only the fact asked about and does not waive other explicit requirements. "
            "Use conventions only where the specification leaves a choice open; do not "
            "invent validation rules or silently drop constraints. "
            "Respect the exact entry point, signature and return shape. If the user gave "
            "a vague or refused answer, do not invent a resolved requirement; use the "
            "prompt and justified conventional defaults. Think through edge cases, "
            "ordering, indexing, mutation, imports and termination before emitting code. "
            "Use standard-library helpers when appropriate and omit top-level examples. "
            "Optionally include a few small assert statements justified by explicit "
            "examples, the actual answer, or unambiguous mathematical properties. "
            "Do not use unstated assumptions or your own implementation's output as "
            "expected results; use an empty checks string if no justified checks exist. "
            "Return only JSON with escaped string newlines: "
            '{"code":"complete Python source","checks":"optional Python asserts"}.',
            {**task, "clarification": clarification},
        )
        best = _code(solution, entry_point)
        checks = _object(solution).get("checks", "")
        failure = (
            "" if best else "Response is not valid Python with the required top-level function."
        )
        if (
            best
            and self.config["sandbox_checks"]
            and isinstance(checks, str)
            and checks.strip()
            and available()
        ):
            try:
                compile(ast.parse(checks), "<checks>", "exec")
            except (SyntaxError, TypeError, ValueError, RecursionError):
                checks = ""
            if checks.strip():
                try:
                    output = env.exec_code(best + "\n\n" + checks + f'\nprint("{_SUCCESS}")\n')
                    if not any(line.strip() == _SUCCESS for line in str(output).splitlines()):
                        failure = "Self-generated checks failed. Sandbox output:\n" + str(output)
                except Exception:
                    pass  # Sandbox infrastructure failure is not a correctness signal.
        if failure:
            repaired = call(
                "Repair the full Python implementation using the task and actual "
                "clarification. For failing self-generated checks, first verify that the "
                "assertion follows from the specification: correct an unjustified check "
                "rather than changing correct behavior to satisfy it. Preserve the "
                "required entry point and signature. Return only the complete code in "
                "a fenced python block.",
                {
                    **task,
                    "clarification": clarification,
                    "candidate": best or solution or "",
                    "checks": checks,
                    "failure": failure,
                },
            )
            best = _code(repaired, entry_point) or best
        if not best:
            best = (
                f"def {entry_point}(*args, **kwargs):\n"
                '    raise RuntimeError("No valid implementation was generated within the budget.")'
            )
        # Cleanup is intentionally last: it changes no model, human, or sandbox
        # interaction. Earlier self-checks can still trigger a repair call.
        return sanitize_implementation(best, entry_point) or "```python\n" + best + "\n```"
