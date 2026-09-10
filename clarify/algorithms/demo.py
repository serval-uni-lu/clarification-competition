# SPDX-FileCopyrightText: 2026 Cedric Richter <cedric.richter@uni.lu>
#
# SPDX-License-Identifier: MIT
"""LLM baseline.

A simple baseline implementation of an LLM-based clarification algorithm.
The LLM is asked for a solution to the coding problem. If it abstains on its
own, the respond is sent as a clarification request to the user. 
Otherwise, the answer of the LLM is returned.

Team: Organizers
Team Members: Amal AKLI, Jie JW Wu, Cedric Richter, Mike Papadakis
Main Contact: cedric.richter@uni.lu
"""

from __future__ import annotations
from typing import Any

from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.baselines import ClarificationAlgorithmBase
from clarify.runtime import _validate_and_parse_evalplus_result

DEFAULT_MBPP_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

Enclose your solution in ```python and ```.
""".strip()


class LLMClarification(ClarificationAlgorithmBase):

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {"role": "user", "content": (
                DEFAULT_MBPP_TEMPLATE
                    .replace("{prompt}", problem["prompt"])
                    .replace("{entry_point}", problem["entry_point"])
            )}
        ]

        response = env.llm(
            messages
        )

        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # The response might not contain an answer, assume a question is raised.
                try:
                    answer = env.ask_human(response)
                    messages += [{"role": "user", "content": answer}]
                    response = env.llm(messages)
                    messages += [{"role": "assistant", "content": response}]
                except TooManyQuestionException:
                    return response


