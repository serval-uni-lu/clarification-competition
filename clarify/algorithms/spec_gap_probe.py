# SPDX-FileCopyrightText: 2026 Zhang Chuang <zhangchuang@lenovo.com>
#
# SPDX-License-Identifier: MIT

"""SpecGapProbe — specification-gap driven clarification.

Instead of generating candidate implementations and looking for behavioural
disagreement (the divergence-probe approach), this algorithm starts from the
requirement text itself: it enumerates every concrete behavioural decision the
prompt *does* specify, then identifies the most important decision it *does not*
specify. A clarifying question is asked only about that single unspecified
decision, phrased in the declarative, slot-shaped style that the oracle's
QUALITY=3 rubric rewards.

Why this architecture:
- The divergence probe is a *proxy* signal — candidates can agree for the
  wrong reason (all three land on the same wrong default), and they can
  disagree for no behavioural reason (purely stylistic variation).
- A spec-gap scan is a *direct* signal — it reads the requirement the way a
  human engineer would, looking for the missing sentence that would make the
  specification complete. This matches how the hidden requirements are
  constructed: they are sentences deleted from the original complete prompt.

Team: zhangchuangnankai
Team Members: Zhang Chuang (Lenovo)
Main Contact: zhangchuang@lenovo.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

SOLVE_TEMPLATE = """Implement the Python function `{entry_point}` described below.

{prompt}

{extra}

Enclose your solution in ```python and ```.
""".strip()

SCAN_TEMPLATE = """You are reading a coding problem requirement. Your job is to identify
every *behavioural decision* that the requirement must pin down for the function to be
fully specified, and to classify each one as either SPECIFIED (the requirement states it
unambiguously) or UNSPECIFIED (the requirement is silent or ambiguous about it).

A behavioural decision is any fact that could change the correct output: default values,
boundary rules, ordering / tie-break rules, return-value conventions, input validation
behaviour, handling of empty / duplicate / None inputs, units, precision, etc.

### Requirement

{prompt}

### Output format

First, list every SPECIFIED decision, one per line, starting with "- SPECIFIED: ".
Use the exact wording from the requirement where possible.

Then list every UNSPECIFIED decision, one per line, starting with "- UNSPECIFIED: ".
For each unspecified decision, briefly note why it matters and what the plausible
alternatives would be.

Be thorough — a correct implementation depends on all of these. But only list decisions
that are genuinely behavioural, not stylistic.
""".strip()

PICK_TEMPLATE = """From the list of unspecified behavioural decisions below, pick the SINGLE
one that is most likely to determine whether the implementation passes its test cases.
That is: pick the gap whose resolution would change the most test outcomes.

### Requirement

{prompt}

### Unspecified decisions

{gaps}

### Output format

Output exactly:

GAP: <the most important unspecified decision, stated as a short fact in the same style
  the requirement would use — e.g. "the initial balance is zero" rather than "what is
  the initial balance?">
QUESTION: <a single question asking for exactly that fact, phrased as a direct
  declarative lookup — "What is the initial balance of the account?" not "How should
  the initial balance be handled?">

If there are no unspecified decisions, or none of them could change test outcomes,
output exactly:

NO_GAP
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

REGEN_EXTRA_TEMPLATE = """You asked the user this clarifying question about the requirement:

Q: {question}

The user answered:

A: {answer}

The decision this question was meant to settle: {gap}

Implement `{entry_point}` taking the exchange into account:
- If the answer states a concrete rule, follow it exactly.
- If the answer is empty, vague, or declines to answer, resolve the decision in the
  way that is most consistent with the requirement's own examples and names, preferring
  the most conventional standard behaviour.""".strip()


class SpecGapProbe(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "generation_attempts": 3,
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

    def _scan_gaps(self, env: ClarificationEnvironment, prompt: str) -> list[str]:
        """Return list of UNSPECIFIED decision strings from the scan pass."""
        response = env.llm(SCAN_TEMPLATE.replace("{prompt}", prompt))
        gaps = []
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.startswith("- UNSPECIFIED:"):
                gaps.append(stripped[len("- UNSPECIFIED:"):].strip())
        return gaps

    def _pick_gap(
        self, env: ClarificationEnvironment, prompt: str, gaps: list[str]
    ) -> tuple[str, str]:
        """Pick the most important gap and formulate a slot-shaped question.

        Returns (gap_statement, question) or ("", "") if no gap worth asking.
        """
        rendered = "\n".join(f"- {g}" for g in gaps)
        response = env.llm(
            PICK_TEMPLATE.replace("{prompt}", prompt).replace("{gaps}", rendered)
        )

        if "NO_GAP" in response:
            return "", ""

        gap = ""
        gap_marker = response.split("GAP:", 1)
        if len(gap_marker) == 2:
            gap = gap_marker[1].split("QUESTION:", 1)[0].strip()

        marker = response.split("QUESTION:", 1)
        if len(marker) < 2:
            return "", ""

        return gap, marker[1].strip()

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

        # Phase 1 — scan the requirement for specification gaps.
        gaps = self._scan_gaps(env, prompt)

        # Phase 2 — if there are no gaps, just solve and return.
        if not gaps or not env.can_ask():
            return self._solve(env, prompt, entry_point)

        # Phase 3 — pick the most impactful gap and formulate a question.
        gap, question = self._pick_gap(env, prompt, gaps)
        if not question:
            return self._solve(env, prompt, entry_point)

        # Phase 4 — audit the question against the five quality criteria.
        if self.config["audit_question"]:
            question = self._audit(env, prompt, question)
            if not question:
                return self._solve(env, prompt, entry_point)

        # Phase 5 — ask and regenerate with the answer grounded.
        try:
            clarification = env.ask_human(question)
        except TooManyQuestionException:
            return self._solve(env, prompt, entry_point)

        extra = (
            REGEN_EXTRA_TEMPLATE.replace("{question}", question)
            .replace("{answer}", clarification or "(no answer provided)")
            .replace("{gap}", gap or "not recorded")
            .replace("{entry_point}", entry_point)
        )
        return self._solve(env, prompt, entry_point, extra=extra)
