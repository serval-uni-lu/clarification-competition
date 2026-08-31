from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from clarify.llm import LanguageModel
from clarify.runtime import EvalPlusDockerInstanceEvaluator

@dataclass(frozen=True)
class ClarificationConfiguration:

    language_model      : str = "gpt-4.1-mini"
    temperature         : float = 0.7
    clarification_model : str | None = None

    # Budget constraints
    max_clarification_turns : int = 1
    max_clarification_budget: float = 1.0
    max_prompt_budget       : float = 1.0

class TooManyQuestionException(Exception):
    pass


class LimitsExceededException(Exception):
    pass


HUMAN_PROMPT = """
The original description of a coding problem is modified so that the requirements become inconsistent, incomplete, or ambiguous. Given the modified description, some clarifying questions were raised to clarify the description.
Given the original and modified problem description, evaluate the quality of the clarifying questions. 
Please provide an integer representing the quality of questions (3: Good questions that recover the modified requirements; 2: Fair questions but they cannot help recover the modified requirements; 1: No questions).
QUALITY=[your int]
Please also provide answers to the clarifying questions to recover the modified requirements in the original problem description compared to the modified one. 
If there are no clarifying questions at all, return empty answers.
ANSWERS=```[your answer]```
Please strictly follow the format QUALITY=[the int] and ANSWERS=```[the answer]``` in the response! Surround your answer with markdown!

### Questions: {clarifying_questions}
### Modified Problem Description: {problem} 
### Original Description: {original_problem}
"""


class _ClarificationEnvironment:

    def __init__(self, 
                 config : ClarificationConfiguration, 
                 problem : dict[str, Any],
                 llm_api_hook : callable = None,
                 clarification_api_hook : callable = None):
        self._config = config
        self._problem = problem

        self._llm = LanguageModel(config.language_model, temperature = config.temperature)
        self._clarify_llm = LanguageModel(
            config.clarification_model or config.language_model,
            temperature = 0.0
        )
        self._num_clarification_turns = 0
        self._clarification_history = []

        self._clarification_hook = clarification_api_hook
        self._llm_hook = llm_api_hook

    @property
    def history(self):
        return list(self._clarification_history)

    @property
    def prompt_cost(self):
        return self._llm.total_cost

    @property
    def prompt_budget(self):
        return self._config.max_prompt_budget

    @property
    def clarification_cost(self):
        return self._clarify_llm.total_cost

    def can_ask(self):
        if self.clarification_cost >= self._config.max_clarification_budget:
            return False

        return self._num_clarification_turns < self._config.max_clarification_turns

    def _parse_human_response(self, completion_content):
        question_quality = re.findall(r'QUALITY\s*=?\s*(\d+)', completion_content)
        answers = re.findall(r'ANSWERS\s*=?\s*```(.+?)```', completion_content, flags=re.DOTALL)
        answer_str = answers[0] if answers else ""
        question_quality_str = question_quality[0] if question_quality else ""
        return question_quality_str, answer_str

    # Main API -----------------------------------------------------------------

    def llm(self, messages : list[dict[str, str]] | str) -> str:
        if self.prompt_cost >= self.prompt_budget:
            raise LimitsExceededException(f"You exceeded the prompt budget.")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if self._llm_hook: self._llm_hook(messages[-1])
        response = self._llm(messages)
        if self._llm_hook: self._llm_hook({"role": "assistant", "content": response})
        return response

    def ask_human(self, query : str) -> str:
        if not self.can_ask():
            raise TooManyQuestionException("You exceeded the clarification budget by asking too many or overly complex questions.")

        human_prompt = HUMAN_PROMPT.replace(
            "{clarifying_questions}", query
        ).replace(
            "{problem}", self._problem.get("prompt", "[REDACTED]")
        ).replace(
            "{original_problem}", self._problem.get("reference_prompt", "[REDACTED]")
        )

        if self._clarification_hook:
            self._clarification_hook({"role": "user", "content": query})

        try:
            response = self._clarify_llm(human_prompt)
            if self._clarification_hook:
                self._clarification_hook({"role": "assistant", "content": response})
            score, answer = self._parse_human_response(
                response
            )
            self._clarification_history.append((query, score, answer))
            return answer
        except Exception:
            return "Cannot answer the given question."
        finally:
            self._num_clarification_turns += 1

    def exec_code(self, complete_code : str) -> str:
        return EvalPlusDockerInstanceEvaluator(
            self._problem, None
        ).exec_code(complete_code)


class ClarificationEnvironment:

    def __init__(self, *args, **kwargs):
        protected_env = _ClarificationEnvironment(*args, **kwargs)
        object.__setattr__(self, "_env", protected_env)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise RuntimeError(f"Access to '{name}' is not permitted")
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        raise RuntimeError("Environment is read-only for participants")