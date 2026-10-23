# SPDX-FileCopyrightText: 2026 XikaiChen <SteveChen19991225@outlook.com>
#
# SPDX-License-Identifier: MIT

"""Registration placeholder; replace with the final clarification algorithm.

This draft makes one SDK model call to produce a Python implementation. It does
not ask a clarification question and has not been evaluated. It exists only to
prepare the registration pull request while the final method is still open.

Team: SteveChen
Team Members: XikaiChen
Main Contact: SteveChen19991225@outlook.com
"""

from clarify.baselines import ClarificationAlgorithmBase


class SteveChenClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {}

    def run(self, env, problem) -> str:
        prompt = (
            f"Implement the Python function `{problem['entry_point']}` for this task:\n\n"
            f"{problem['prompt']}\n\n"
            "Return a self-contained implementation in a ```python code block."
        )
        return env.llm([{"role": "user", "content": prompt}])
