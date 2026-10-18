# SPDX-FileCopyrightText: 2026 Shane Califano <scalifano4740@floridapoly.edu>
#
# SPDX-License-Identifier: MIT

"""ReqClarify

A clarification algorithm.

Team: PolyTeam
Team Members: Shane Califano, Mohamed Sylla, William Blaine
Main Contact: scalifano4740@floridapoly.edu
"""

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment


class ReqClarify(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        """Initial baseline placeholder for registration."""
        messages = [{"role": "user", "content": problem["prompt"]}]
        return env.llm(messages)
