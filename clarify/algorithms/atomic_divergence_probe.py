# SPDX-FileCopyrightText: 2026 Zhang Chuang <zhangchuang@lenovo.com>
#
# SPDX-License-Identifier: MIT

"""Atomic Divergence Probe.

Independent implementations of the same requirement are generated under
deliberately different readings of it. Their behaviour is then compared: only
when they disagree on a concrete behavioural decision does the algorithm spend
its clarification turn, asking for that single fact. Candidate questions are
audited against the five quality criteria (criticality, search space, leakage,
atomicity, objectivity) before being sent, and the final implementation is
regenerated with the clarification in context.

The gate matters as much as the question: when the readings agree, the
algorithm stays silent and returns the first implementation, keeping the
turn-discount penalty at zero for already-specified requirements.

Team: zhangchuangnankai
Team Members: Zhang Chuang (Lenovo)
Main Contact: zhangchuang@lenovo.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

# Each lens pushes the model towards a different resolution of whatever the
# requirement leaves open, which is what makes the candidates diverge.
READING_LENSES = [
    "Follow the most literal, minimal reading of the requirement.",
    (
        "Follow the reading a defensive engineer would choose where the requirement is "
        "silent: handle empty inputs, duplicates, and boundary values explicitly."
    ),
    (
        "Follow the reading that is most internally consistent with any examples, names, "
        "or hints in the requirement, even where it differs from the obvious default."
    ),
]

SOLVE_TEMPLATE = """Implement the Python function `{entry_point}` described below.

{prompt}

{extra}

Enclose your solution in ```python and ```.
""".strip()

COMPARE_TEMPLATE = """Several independent implementations of the same requirement were produced.

### Requirement

{prompt}

{candidates}

### Task

Decide whether these implementations disagree on a *behavioural* decision, i.e. whether
there exist inputs on which they return different results, or whether they disagree about
a rule the requirement never states.

If they agree behaviourally, output exactly:

DIVERGENCE: NO

If they disagree, identify the SINGLE most consequential disagreement and state the one
concrete fact that would settle it. That fact must be:
- a real blocker for correctness, not style, naming, performance or code structure;
- impossible to infer reliably from the requirement text alone;
- exactly one fact: a constant, a default, a boundary rule, an ordering or tie-break rule,
  or a return convention;
- resolvable by a single unambiguous answer.

Output exactly:

DIVERGENCE: YES
FACT: <the one fact, phrased as a statement of what is currently unknown>
QUESTION: <one question asking for exactly that fact>
""".strip()

AUDIT_TEMPLATE = """Rewrite the question below so that it satisfies every one of the five criteria.

CRITICALITY: it addresses a real blocker to a correct solution, not a cosmetic detail.
SEARCH SPACE: its answer cannot be guessed or inferred from the requirement, so it must
come from the user.
LEAKAGE: a full answer is a specific fact needed to implement the function, not the
reference implementation, its algorithm structure, or the test cases.
ATOMICITY: a complete and precise answer consists of exactly one fact. A question fails
if it bundles sub-questions, and also if it is broad enough that answering it precisely
would require revealing more than one independent fact.
OBJECTIVITY: it admits a single unambiguous answer, not a matter of taste.

### Requirement

{prompt}

### Question

{question}

If the question cannot be made to satisfy all five criteria, output exactly:

NO_QUESTIONS

Otherwise output exactly one line:

QUESTION: <the rewritten question, asking for exactly one fact>
""".strip()

REGEN_EXTRA_TEMPLATE = """The user clarified the following:

{clarification}

Implement `{entry_point}` so that it follows this clarification exactly.""".strip()


class AtomicDivergenceProbe(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # number of independent implementations generated before deciding to ask
        "num_candidates": 3,
        # how often a malformed (unfenced) model answer is retried
        "generation_attempts": 3,
        # audit and rewrite the question against the five criteria before sending it
        "audit_question": True,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__({**self.DEFAULT_CONFIG, **(config or {})})

    def _solve(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str, extra: str = ""
    ) -> str:
        message = SOLVE_TEMPLATE.replace("{prompt}", prompt).replace(
            "{entry_point}", entry_point
        ).replace("{extra}", extra)
        messages = [{"role": "user", "content": message}]

        response = env.llm(messages)
        messages.append({"role": "assistant", "content": response})

        for _ in range(self.config["generation_attempts"]):
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError as error:
                messages.append({"role": "user", "content": str(error)})
                response = env.llm(messages)
                messages.append({"role": "assistant", "content": response})

        return response

    def _collect_candidates(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str
    ) -> list[str]:
        count = max(1, self.config["num_candidates"])
        lenses = [READING_LENSES[i % len(READING_LENSES)] for i in range(count)]
        return [self._solve(env, prompt, entry_point, extra=lens) for lens in lenses]

    def _compare(
        self, env: ClarificationEnvironment, prompt: str, candidates: list[str]
    ) -> tuple[bool, str]:
        rendered = "\n\n".join(
            f"### Implementation {index}\n\n```python\n{code}\n```"
            for index, code in enumerate(candidates)
        )
        response = env.llm(
            COMPARE_TEMPLATE.replace("{prompt}", prompt).replace("{candidates}", rendered)
        )

        head = response.split("DIVERGENCE:", 1)
        if len(head) < 2 or not head[1].lstrip().upper().startswith("YES"):
            return False, ""

        marker = response.split("QUESTION:", 1)
        if len(marker) < 2:
            return False, ""

        return True, marker[1].strip()

    def _audit(self, env: ClarificationEnvironment, prompt: str, question: str) -> str:
        response = env.llm(
            AUDIT_TEMPLATE.replace("{prompt}", prompt).replace("{question}", question)
        )

        if "NO_QUESTIONS" in response:
            return ""

        marker = response.split("QUESTION:", 1)
        if len(marker) < 2:
            return ""

        return marker[1].strip().strip("`").strip()

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        candidates = self._collect_candidates(env, prompt, entry_point)

        if not env.can_ask():
            return candidates[0]

        diverged, question = self._compare(env, prompt, candidates)
        if not diverged:
            return candidates[0]

        if self.config["audit_question"]:
            question = self._audit(env, prompt, question)
            if not question:
                return candidates[0]

        try:
            clarification = env.ask_human(question)
        except TooManyQuestionException:
            return candidates[0]

        return self._solve(
            env,
            prompt,
            entry_point,
            extra=REGEN_EXTRA_TEMPLATE.replace("{clarification}", clarification).replace(
                "{entry_point}", entry_point
            ),
        )