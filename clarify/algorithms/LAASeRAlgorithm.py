# SPDX-FileCopyrightText: 2026 Matias Brizzio <matias.brizzio@list.lu>
#
# SPDX-License-Identifier: MIT

"""Algorithm name.

LAASeR (Bribery) algorithm.

We make the LLM aware that its output will be part of a competition.
We explain to it that the main goal is to implement the correct function,
but unnecessary questions are penalized.


Team: LAASeR - LIST
Team Members: Matias Brizzio, Jordi Cabot, Renzo Degiovanni
Main Contact: matias.brizzio@list.lu
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, LimitsExceededException, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

READY_TO_CODE_KEY = "READY_TO_CODE"
QUESTION_KEY = "QUESTION"

READY_TO_CODE_TEMPLATE = """
Please find below a coding task.
Task (function to implement: `{entry_point}`):
`{prompt}`

Important: your final function must be named exactly `{entry_point}`, even if
the task text above uses a different (e.g. generic or placeholder) name for
it. The hidden tests call the function `{entry_point}` by that exact name, so
any other name fails every test regardless of whether the logic is correct.

Please review it and determine if you have everything to implement it,
and write READY_TO_CODE followed by ```python and ```

If it is impossible for you to implement it because the description lacks crucial details, 
please ask a simple and concrete question to the user (QUESTION: ...)

Notice that you will be participating in a prompt clarification competition, 
where the goal is to implement the correct python function. 
Clarification questions are just needed on prompts that are inaccurate, 
but unnecessary questions are penalized. 
""".strip()


class LAASeRAlgorithm(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:

        clarifications = []
        candidate = ""

        while True:
            prompt = problem["prompt"]
            if len(clarifications) > 0:
                prompt += "Clarifications:\n"
                for c in clarifications:
                    prompt += c

            messages = [
                {
                    "role": "user",
                    "content": (
                        READY_TO_CODE_TEMPLATE.replace("{prompt}", prompt).replace(
                            "{entry_point}", problem["entry_point"]
                        )
                    ),
                }
            ]
            try:
                response = env.llm(messages)
                if READY_TO_CODE_KEY in response:
                    candidate = _validate_and_parse_evalplus_result(response)
                    return candidate
                elif QUESTION_KEY in response:
                    _, clarifying_question = response.split(QUESTION_KEY, 1)
                    clarifying_question = (
                        clarifying_question[1:]
                        if clarifying_question.startswith(":")
                        else clarifying_question
                    )
                    clarification = env.ask_human(clarifying_question)
                    num_rounds = len(clarifications)
                    clarifications += [
                        f"Question #{num_rounds + 1}:\n{clarifying_question}\nAnswer:\n{clarification}\n"
                    ]

                else:
                    return candidate
            except TooManyQuestionException:
                print("---> LAASeR: TooManyQuestionException")
                return candidate
            except LimitsExceededException:
                print("---> LAASeR: LimitsExceededException")
                return candidate
            except Exception:
                # The response might not contain an answer, assume a question is raised.
                print("---> LAASeR: Exception")
                return candidate
