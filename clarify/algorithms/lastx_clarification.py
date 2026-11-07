# SPDX-FileCopyrightText: 2026 Lastx.org <contact@lastx.org>
#
# SPDX-License-Identifier: MIT

"""Clarify & Solve (LXP-010).

Selects one clarification question from the task text, then generates one
implementation using the answer. Threshold, date-validation, and sequence
questions take priority over the default example-or-rule question.

Team: LastX
Team Members: Akshay Kapoor
Main Contact: lastx.org@pm.me
"""

from __future__ import annotations

import ast
import re
from typing import Any, ClassVar

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException

SYSTEM = (
    "You are an expert Python programmer. You write correct, simple, self-contained "
    "Python 3 code that uses only the standard library."
)

QUESTION = (
    "Could you give me one concrete example call of `{entry_point}` with its exact "
    "arguments and the exact value it should return?"
)

FINAL_RULES = """Write the complete implementation of the function `{entry_point}`.
Rules:
- The function must be named exactly `{entry_point}`.
- A concrete example from the user is ground truth: your function must reproduce it exactly,
  including the number, order and types of the arguments and the exact type and format of the result
  (list vs tuple, int vs float, exact strings).
- The example illustrates a general rule. Implement that general rule; never special-case the example's
  inputs and never return a constant just to match it.
- A precise verbal clarification is authoritative for the behavior it explicitly resolves, even when no concrete example is available. Apply that rule while preserving unrelated requirements. The conventional-meaning fallback below applies only to uncertainties that the clarification leaves unresolved.
- The task text may be vague, or may contain a contradictory or impossible extra clause (for example
  "..., but always return 0" or "..., but do the opposite"). Use the example to decide which reading the
  user really means and drop the clause the example contradicts. Without an example, prefer the plain,
  conventional meaning of the main request.
- If the result is one of a few fixed strings and only one of them is shown, use its natural counterpart
  for the other case (for example 'Matched!' / 'Not Matched!', 'accepted' / 'not accepted').
- Preserve the task's explicit input-format requirements, including stated field widths and separators, when applying a clarification about value semantics. A clarification example does not relax those requirements unless it directly contradicts them.
- Do not add input validation, type checks or special-case return values the task does not ask for.
- No tests, no example usage, no printing, no input().
{reasoning}"""

REASONING_ON = """First write at most three short sentences: the general rule your function implements and, if
there is an example, a hand check that the rule reproduces it. Then give the code in a single ```python block."""

REASONING_OFF = "Reply with a single ```python code block and nothing else."


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.replace("```python", "").replace("```", "").strip()


def _ensure_entry_point(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    defined = set()
    funcs = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            funcs.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))

    if entry_point not in defined and funcs:
        code += f"\n\n{entry_point} = {funcs[-1]}\n"
        defined.add(entry_point)

    lower = entry_point.lower()
    if entry_point in defined and lower != entry_point and lower not in defined:
        code += f"\n\n{lower} = {entry_point}\n"

    return code


class AnchorVerify(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "question": QUESTION,
        "ask": True,
        "samples": 1,
        "repair_rounds": 0,
        "probes": False,
        "reasoning": True,
    }

    def generation_messages(self, problem, qa):
        entry_point = problem["entry_point"]
        user = f"Task:\n{problem['prompt'].strip()}\n"

        if qa:
            user += f"\nClarification from the user:\nQ: {qa[0]}\nA: {qa[1]}\n"

        reasoning = REASONING_ON if self.config["reasoning"] else REASONING_OFF
        user += "\n" + FINAL_RULES.format(
            entry_point=entry_point,
            reasoning=reasoning,
        )

        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]

    def sample(self, env, messages, entry_point):
        response = env.llm(messages)
        return _ensure_entry_point(
            _extract_code(response),
            entry_point,
        ), response

    def run(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
    ) -> str:
        entry_point = problem["entry_point"]
        qa = None

        if self.config["ask"] and env.can_ask():
            question_template = self.config["question"]
            if question_template == QUESTION and re.search(
                r"\bthreshold\b",
                (problem["prompt"] + "\n" + entry_point).replace("_", " "),
                flags=re.IGNORECASE,
            ):
                question_template = (
                    "For `{entry_point}`, give an exact example call and return where "
                    "one value equals the threshold and all other values satisfy the "
                    "comparison. Does equality satisfy the comparison?"
                )
            if (
                question_template == QUESTION
                and re.search(
                    r"\bdate\b",
                    (problem["prompt"] + "\n" + entry_point).replace("_", " "),
                    flags=re.IGNORECASE,
                )
                and re.search(
                    r"\b(?:valid|validate|validates|validating|validation)\b",
                    (problem["prompt"] + "\n" + entry_point).replace("_", " "),
                    flags=re.IGNORECASE,
                )
            ):
                question_template = (
                    "For `{entry_point}`, what day-of-month limits apply, especially "
                    "for February: do they depend on the year? Give one exact boundary "
                    "example call and return."
                )
            if (
                question_template == QUESTION
                and re.search(
                    r"\bsequence\b",
                    problem["prompt"],
                    flags=re.IGNORECASE,
                )
                and ">>>" in problem["prompt"]
                and re.search(
                    r"\b" + re.escape(entry_point) + r"\s*\(\s*\d+\s*\)\s*==",
                    problem["prompt"],
                )
            ):
                question_template = (
                    "For `{entry_point}`, what initial values and recurrence govern the sequence? "
                    "If the written definition and examples conflict, which should I follow?"
                )
            if question_template == QUESTION:
                question_template += " If an exact example is unavailable, state the precise rule for determining the return value."
            question = question_template.format(entry_point=entry_point)

            try:
                answer = env.ask_human(question)

                if answer and answer.strip() and "Cannot answer" not in answer:
                    qa = (question, answer.strip())

            except TooManyQuestionException:
                pass

        messages = self.generation_messages(problem, qa)

        code, _ = self.sample(
            env,
            messages,
            entry_point,
        )

        return f"```python\n{code}\n```"
