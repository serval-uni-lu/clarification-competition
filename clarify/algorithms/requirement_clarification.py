# SPDX-FileCopyrightText: 2026 Junjie Shi, Yiran Zhang, Weisong Sun <chunkit001@e.ntu.edu.sg>
#
# SPDX-License-Identifier: MIT

"""RequirementClarification.

Initial implementation for draft registration. Ask one targeted question when
the model identifies a missing requirement; otherwise return the implementation.

Team: ntu.csl
Team Members: Junjie Shi, Yiran Zhang, Weisong Sun
Main Contact: Junjie Shi <chunkit001@e.ntu.edu.sg>
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException


class RequirementClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Implement the requested Python function. If a missing or ambiguous "
                    "requirement prevents a reliable implementation, ask one targeted "
                    "question about that requirement, with no code. Otherwise return "
                    "a self-contained implementation in a fenced python code block. "
                    "Do not ask for hidden tests or a reference implementation."
                ),
            },
            {
                "role": "user",
                "content": f"Entry point: {problem['entry_point']}\n\n{problem['prompt']}",
            },
        ]
        response = env.llm(messages)
        if "```python" in response:
            return response

        messages.append({"role": "assistant", "content": response})
        if env.can_ask():
            try:
                answer = env.ask_human(response)
            except TooManyQuestionException:
                answer = "No further clarification is available."
            messages.append({"role": "user", "content": answer})

        messages.append(
            {
                "role": "user",
                "content": (
                    "Return the best implementation supported by the available "
                    "requirements in a fenced python code block. No further questions."
                ),
            }
        )
        return env.llm(messages)
