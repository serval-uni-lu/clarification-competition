# SPDX-FileCopyrightText: 2026 Khalid Tounoussi <khalid.tounoussi.1@ens.etsmtl.ca>
#
# SPDX-License-Identifier: MIT

"""Gated clarification.

Decides whether the task is genuinely underspecified and, if so, asks one atomic
question before writing the implementation. The clarification is answered by the
user and the solution is then generated fresh with that answer in context.

Generating fresh matters. An earlier variant kept a seed candidate and asked the
model to revise it; on the validation split that cost 9 points of Pass@1 on
underspecified tasks, because showing the model its own first attempt anchors it to
the interpretation it already committed to. A third of this benchmark contradicts
the user's real intent on purpose, so the clarification has to be able to overturn
the task description outright.

Team: STIL-ETS
Team Members: Khalid Tounoussi
Main Contact: khalid.tounoussi.1@ens.etsmtl.ca
"""

from __future__ import annotations

import ast
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

GATE_TEMPLATE = """
You are reviewing a coding task before implementing it.

Task (function to implement: `{entry_point}`):
{prompt}

Decide whether the task is underspecified: is there a fact you would have to guess,
where guessing wrong would make a correct-looking implementation fail the user's tests?

Ignore stylistic choices, naming, and anything you can reasonably infer from the
description or examples. Only a genuine behavioural blocker counts.

If everything needed is present, reply with exactly:
NO_QUESTION

Otherwise reply with exactly one line:
QUESTION: <a single, specific question about exactly one missing fact>

The question must target one fact only, admit one unambiguous answer, and must not
ask for the algorithm, the implementation, or the test cases.
""".strip()

SOLVE_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves
the following problem:

{prompt}
{clarification}
Enclose your solution in ```python and ```.
""".strip()

CLARIFICATION_BLOCK = """
The user was asked a clarifying question and answered:
Q: {question}
A: {answer}

Follow that answer exactly, even where it contradicts your first reading of the task.
"""

EMPTY_SOLUTION = """```python
def {entry_point}(*args, **kwargs):
    raise NotImplementedError
```"""

REPAIR_TEMPLATE = """
Your response did not contain a valid fenced Python solution.
Return only the complete implementation of `{entry_point}`, enclosed in ```python and ```.
""".strip()


def usable_solution(response: str, entry_point: str) -> bool:
    """True when the response carries a fenced block that is real, runnable code.

    A fence alone is not enough: an empty block and a syntax error both parse out of
    the fence happily, and either one would silently pass for a solution.
    """
    try:
        code = _validate_and_parse_evalplus_result(response)
    except ValueError:
        return False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in ast.walk(tree)
    )


def complete_code(
    env: ClarificationEnvironment,
    prompt: str,
    entry_point: str,
    repair_attempts: int,
) -> str:
    """Ask for an implementation, nudging the model until the fence parses.

    Always returns a string, so an exhausted budget can never hand the evaluator None.
    """
    messages = [{"role": "user", "content": prompt}]
    last = ""

    for attempt in range(repair_attempts + 1):
        try:
            response = env.llm(messages)
        except LimitsExceededException:
            break

        last = response
        if usable_solution(response, entry_point):
            return response

        if attempt == repair_attempts:
            break

        messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": REPAIR_TEMPLATE.format(entry_point=entry_point)},
        ]

    return last if last else EMPTY_SOLUTION.format(entry_point=entry_point)


def parse_question(verdict: str, ask_when_unsure: bool) -> str | None:
    """Extract the single question from a gate verdict, or None.

    Model prose is never forwarded to the user verbatim: an unrecognisable verdict
    falls back to ``ask_when_unsure`` rather than sending whatever was produced.
    """
    lines = [line.strip() for line in verdict.splitlines() if line.strip()]

    if any(line.upper().strip("`*. ") == "NO_QUESTION" for line in lines):
        return None

    questions = [
        line.split(":", 1)[1].strip()
        for line in lines
        if line.upper().lstrip("`*- ").startswith("QUESTION:")
    ]

    if len(questions) == 1 and questions[0]:
        return questions[0]

    if ask_when_unsure and len(lines) == 1 and lines[0].endswith("?"):
        return lines[0]

    return None


class GatedClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # Retries allowed when a reply does not contain a usable ```python block.
        "repair_attempts": 2,
        # Treat an unrecognisable gate verdict as a question when it reads like one.
        # Asking costs 4% of TDS; staying silent on an underspecified task costs the
        # whole task, so the default leans towards asking.
        "ask_when_unsure": True,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]
        clarification = ""

        if env.can_ask():
            try:
                verdict = env.llm(
                    GATE_TEMPLATE.format(prompt=problem["prompt"], entry_point=entry_point)
                )
            except LimitsExceededException:
                verdict = ""

            question = parse_question(verdict, self.config["ask_when_unsure"])
            if question:
                try:
                    answer = env.ask_human(question)
                    clarification = CLARIFICATION_BLOCK.format(question=question, answer=answer)
                except TooManyQuestionException:
                    pass

        return complete_code(
            env,
            SOLVE_TEMPLATE.format(
                prompt=problem["prompt"],
                entry_point=entry_point,
                clarification=clarification,
            ),
            entry_point,
            self.config["repair_attempts"],
        )
