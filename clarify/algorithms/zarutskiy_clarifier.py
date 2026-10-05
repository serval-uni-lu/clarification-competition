# SPDX-FileCopyrightText: 2026 Semyon Zarutskiy <zarutskiysy@gmail.com>
#
# SPDX-License-Identifier: MIT

"""ZarutskiyClarifier.

Registration placeholder. The LLM is asked for a solution directly; if it
abstains on its own, its response is sent to the user as a clarifying
question. The final algorithm will be pushed before the submission deadline.

Team: zarutskiysy
Team Members: Semyon Zarutskiy
Main Contact: zarutskiysy@gmail.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

PROMPT_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

Enclose your solution in ```python and ```.
""".strip()


class ZarutskiyClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        content = PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
            "{entry_point}", problem["entry_point"]
        )
        messages = [{"role": "user", "content": content}]

        response = env.llm(messages)
        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # No implementation in the response: treat it as a clarifying question.
                try:
                    answer = env.ask_human(response)
                except TooManyQuestionException:
                    return response
                messages += [{"role": "user", "content": answer}]
                response = env.llm(messages)
                messages += [{"role": "assistant", "content": response}]
