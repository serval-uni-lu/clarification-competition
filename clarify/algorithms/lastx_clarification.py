# SPDX-FileCopyrightText: 2026 LastX Team <lastx.org@pm.me>
#
# SPDX-License-Identifier: MIT

"""LastX Clarification — draft registration placeholder.

This placeholder requests an implementation of the public problem through the
SDK. The final clarification policy and evaluation are pending. No competition
performance is claimed for this placeholder.

Team: LastX
Team Members: Akshay Kapoor, Harveen Singh
Main Contact: lastx.org@pm.me
"""

from __future__ import annotations

from clarify.baselines import ClarificationAlgorithmBase


class LastXClarification(ClarificationAlgorithmBase):
    """Registration placeholder to be replaced before final submission."""

    DEFAULT_CONFIG = {}

    def run(self, env, problem) -> str:
        prompt = (
            "Implement the requested Python function. Return a self-contained "
            "implementation as Python source or a fenced Python block.\n\n"
            f"Entry point: {problem['entry_point']}\n\n{problem['prompt']}"
        )
        return env.llm(prompt)
