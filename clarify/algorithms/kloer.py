# SPDX-FileCopyrightText: 2026 Guillaume Haben <guillaume.haben@telecomnancy.net>
#
# SPDX-License-Identifier: MIT

"""Kloer.

Placeholder submission used to register the team. The LLM is asked for a
solution to the coding problem; if it does not return an implementation, its
response is forwarded to the user as a clarifying question. The final
algorithm will replace this file before the submission deadline.

Team: Kloer
Team Members: Milos Ojdanic, Guillaume Haben, Sami Lazreg
Main Contact: guillaume.haben@telecomnancy.net
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

CODE_PROMPT_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

If the problem is unclear, you may instead ask one clarifying question.
Otherwise, enclose your solution in ```python and ```.
""".strip()


class Kloer(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": CODE_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
                    "{entry_point}", problem["entry_point"]
                ),
            }
        ]
        response = env.llm(messages)
        messages.append({"role": "assistant", "content": response})

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # No implementation found: treat the response as a clarifying question.
                try:
                    answer = env.ask_human(response)
                except TooManyQuestionException:
                    return response
                messages.append({"role": "user", "content": answer})
                response = env.llm(messages)
                messages.append({"role": "assistant", "content": response})
