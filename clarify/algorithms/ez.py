# SPDX-FileCopyrightText: 2026 Evgeny Zotov <evgeny.zotov@uni.lu>
#
# SPDX-License-Identifier: MIT

"""EZ Clarification.

Minimal registration submission for the Clarification Challenge.
Direct LLM generation with clarification fallback: if the model response does
not parse as a valid implementation, it is interpreted as a clarifying
question and posed to the (simulated) user; the clarified specification is
fed back until a valid implementation is produced or the question budget is
exhausted.

This file intentionally mirrors the SDK Quick Start algorithm so the
registration PR is minimal, obviously valid, and easy for organizers to
validate. Improved algorithms (M1+) replace the body in later PRs before the
submission deadline.

Team: EZ
Team Members: Evgeny Zotov
Main Contact: evgeny.zotov@uni.lu
"""

from __future__ import annotations

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result


class EZClarification(ClarificationAlgorithmBase):
    """Direct generation with clarification fallback (SDK Quick Start pattern)."""

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages: list[dict[str, str]] = [{"role": "user", "content": problem["prompt"]}]

        response = env.llm(messages)
        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # The response might not contain an answer; assume a question is raised.
                try:
                    answer = env.ask_human(response)
                    messages += [{"role": "user", "content": answer}]
                    response = env.llm(messages)
                    messages += [{"role": "assistant", "content": response}]
                except TooManyQuestionException:
                    return response
