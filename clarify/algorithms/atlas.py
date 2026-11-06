# SPDX-FileCopyrightText: 2026 YOUR NAME <your.email@example.com>
#
# SPDX-License-Identifier: MIT

"""Atlas.

Registration entry. A simple baseline-style implementation that will be
replaced by the full system before the submission deadline.

Team: Hougar
Team Members: MOHAMMED ISSAM DAOUD BEN MESSAOUD
Main Contact: issammohameddaoud@gmail.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

Enclose your solution in ```python and ```.
""".strip()


class Atlas(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
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
                try:
                    answer = env.ask_human(response)
                    messages.append({"role": "user", "content": answer})
                    response = env.llm(messages)
                    messages.append({"role": "assistant", "content": response})
                except TooManyQuestionException:
                    return response
