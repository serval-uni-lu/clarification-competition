# SPDX-License-Identifier: MIT
"""Team: Ioanna Vougiatzi
Team Members: Ioanna Vougiatzi
Main Contact: vougiatzi.joanna@gmail.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result


class Clarification(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [{"role": "user", "content": problem["prompt"]}]
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
