# SPDX-FileCopyrightText: 2026 Ahmed Nusayer Ashik <ashikan@myumanitoba.ca>
#
# SPDX-License-Identifier: MIT

"""ClarifyBeforeUGo

This algorithm tries to improve code generation by treating requirements analysis as a separate phase:
identify specification defects first, resolve them through iterative human clarification,
and only then generate the implementation.
If issues are found, an atomic clarifying question is asked for the most
critical one. After clarification (or when no issues remain), a solution
is generated from the clarified specification.

Team: ClarifyBeforeUGo
Team Members: Ahmed Nusayer Ashik
Main Contact: ashikan@myumanitoba.ca
"""

from __future__ import annotations

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

ANALYSIS_PROMPT_TEMPLATE = """
You are an expert requirement analyst for coding problems. Analyze the following problem specification for underspecification issues BEFORE any code is written.

### Problem:

{prompt}

### Analysis Instructions:

Examine the specification for the following three categories of underspecification:

1. **Ambiguity**: The description allows multiple valid interpretations. Key phrases could mean different things (e.g., "closer to or larger than" — which one?). Look for vague qualifiers, or terms with multiple technical meanings.
2. **Incompleteness**: Critical information required to determine the intended behavior is missing or insufficiently defined. This includes: missing return types, unspecified edge cases (empty input, negative numbers, None values), truncated or partial descriptions, missing parameter constraints, or unspecified behavior for boundary conditions.
3. **Contradiction**: The specification contains conflicting statements. For example, the docstring says to return a string but the type hint says bool, or two sentences describe opposite behavior for the same case.

### Output Format:

If you find a clear underspecification issue, respond with:
```
CATEGORY: <one of: ambiguity, incompleteness, contradiction>
ISSUE: <one-sentence description of the specific issue found>
QUESTION: <a single, atomic clarifying question that addresses exactly this issue>
```

The question MUST be:
- **Atomic**: Target exactly one specific requirement, not multiple.
- **Non-guessable**: The answer cannot be reliably inferred from context.
- **Requirement-focused**: Do not ask about implementation details, algorithms, or test cases.
- **Objective**: The question should admit a single, unambiguous answer.

If the specification is sufficiently clear and complete for a correct implementation, respond with exactly:
NO_ISSUES
""".strip()

CODE_PROMPT_TEMPLATE = """
Generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`.

{prompt}

Enclose your solution in ```python and ```.
""".strip()

REGEN_CODE_PROMPT_TEMPLATE = """
{prompt}

### Clarifications:
{clarification}

Given the problem and the clarifications above, generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`.

Enclose your solution in ```python and ```.
""".strip()


class PreemptiveClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {"generation_attempts": 3}

    def analyze_prompt(
        self, env: ClarificationEnvironment, prompt: str
    ) -> tuple[str | None, str | None]:
        """Analyze the prompt for underspecification issues before generating code.

        Returns (category, question) if an issue is found, or (None, None) if the prompt is clear.
        """
        response = env.llm(ANALYSIS_PROMPT_TEMPLATE.replace("{prompt}", prompt))

        if "NO_ISSUES" in response:
            return None, None

        # Parse the structured response
        question = None
        category = None
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip()
            elif line.startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip()

        if question:
            return category, question

        return None, None

    def generate_solution(
        self, env: ClarificationEnvironment, problem: dict[str, str], clarifications: list[str]
    ) -> str:
        if clarifications:
            content = (
                REGEN_CODE_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
                .replace("{clarification}", "\n".join(clarifications))
                .replace("{entry_point}", problem["entry_point"])
            )
        else:
            content = CODE_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
                "{entry_point}", problem["entry_point"]
            )

        messages = [{"role": "user", "content": content}]
        response = env.llm(messages)
        messages.append({"role": "assistant", "content": response})

        for _ in range(self.config.get("generation_attempts", 3)):
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError as e:
                messages.append({"role": "user", "content": str(e)})
                response = env.llm(messages)
                messages.append({"role": "assistant", "content": response})

        return response

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        clarifications = []

        # Iteratively analyze and clarify before generating any code
        while env.can_ask():
            category, question = self.analyze_prompt(env, problem["prompt"])

            if question is None:
                break

            try:
                answer = env.ask_human(question)
                clarifications.append(f"Q ({category}): {question}\nA: {answer}\n")
            except TooManyQuestionException:
                break

        # Generate the final solution with all gathered clarifications
        return self.generate_solution(env, problem, clarifications)
