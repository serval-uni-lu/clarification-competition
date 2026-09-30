# SPDX-FileCopyrightText: 2026 Grigoris Alexandrou, George Flourakis <georgeflourakis@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Structured Clarifier.

Decides, per task, whether one clarifying question is worth asking before implementing the
requested function, then generates and verifies the implementation.

Pipeline:
1. Generate a plain seed candidate (no tests) - something concrete for the judge to reason about
   instead of bare prose.
2. If the requirement carries its own `>>>` examples, run them against the seed (no LLM call) and
   hand any failure to the judge as evidence.
3. Judge the requirement against four dimensions - Algorithmic Logic, Constraints, Prompt Defects,
   Input/Output - each marked STATED or UNSTATED. Ask about the first UNSTATED one; ask nothing
   only when all four are STATED.
4. When the requirement has neither a worked example nor a typed function signature (common in
   Mbpp's terse one-line prompts), Input/Output is checked first instead of last, since nothing
   else anchors exact parameter or return shape in that case.
5. Generate the final implementation from the original requirement plus the question and its
   answer only - the seed candidate from step 1 is discarded, so the final model isn't anchored to
   a guess it already committed to.
6. Self-test and repair: the model writes the requirement's own worked examples plus 3-5 of its
   own boundary-case asserts, runs them in Docker, and gets up to two more attempts with the
   traceback fed back on failure.

Team: BALabers
Team Members: Grigoris Alexandrou, George Flourakis
Main Contact: georgeflourakis@gmail.com
"""

from __future__ import annotations

import ast
import doctest
import re
from dataclasses import dataclass, field
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, LimitsExceededException, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

# Used once per round, to get something concrete to judge: a plain implementation, no self-tests.
SEED_CODE_PROMPT = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

The function must be named exactly `{entry_point}`.
Return only the code the solution needs (imports, constants, helpers and `{entry_point}`), with no example usage, prints or tests.

Enclose your solution in ```python and ```.
""".strip()

# _parse_judge_response below, which depends on that exact format.
JUDGE_PROMPT = """
You are deciding what to ask, if anything, before the Python function `{entry_point}` is implemented. Asking is cheap and a wrong guess is costly, so ask unless the requirement leaves nothing to guess.

Requirement:

{prompt}

A candidate implementation:

```python
{candidate}
```
{evidence_block}
Evaluate the requirement against these 4 dimensions in strict order. To prevent the illusion of clarity, for each dimension you MUST brainstorm a "Primary Interpretation" and a "Plausible Alternative Interpretation" that would change the output.
If the prompt and examples do not explicitly rule out the Plausible Alternative, you MUST mark it UNSTATED. Format your analysis exactly like this:

1. Algorithmic Logic (math rules, core operations, ambiguous semantics)
- Primary: <your primary interpretation>
- Alternative: <a mathematically/logically different interpretation that is equally plausible given the vague text>
- Status: STATED or UNSTATED

2. Constraints (edge cases, zeroes, negatives, all-bits-set; DO NOT waste this on empty inputs)
- Primary: <your primary interpretation>
- Alternative: <a different edge-case behavior>
- Status: STATED or UNSTATED

3. Prompt Defects (contradictions between text and examples)
- Primary: <your primary interpretation>
- Alternative: <a contradictory interpretation based on examples>
- Status: STATED or UNSTATED

4. Input/Output (exact types and formats; rely on Python's duck typing where possible)
- Primary: <your primary interpretation>
- Alternative: <a different parameter shape, or a different return type/order>
- Status: STATED or UNSTATED
{priority_override}
Decision:
Priority order is 1, 2, 3, 4, unless the IMPORTANT note above (if present) puts 4 first instead. Find the FIRST dimension in that priority order that is UNSTATED. You MUST ask about the unstated fact from this highest-priority dimension. Do not skip to a lower-priority dimension.
If every dimension is STATED, your final line is NO_QUESTION.

End your response with exactly one line, and write nothing after it:
QUESTION: <exactly ONE atomic, objective question about ONE specific unstated fact. NEVER use "and" to combine questions. NEVER ask for two things. Its answer cannot be inferred from the requirement>
or
NO_QUESTION
""".strip()

# Inserted into JUDGE_PROMPT only when the requirement has neither a worked `>>>` example NOR a
# typed `def` signature: without either, nothing anchors exact parameter/return shape, so
# Input/Output (normally 4th) must be checked first instead, and defaults to UNSTATED unless the
# prose spells the shape out explicitly.
_NO_SHAPE_SIGNAL_PRIORITY_OVERRIDE = """
IMPORTANT: the requirement above gives no worked example (no `>>>` call) and no typed function signature (no type hints in a `def` line). Without either, exact parameter shape (e.g. a single bound vs a full collection) and exact return type/order (e.g. list vs tuple, which side comes first) cannot be safely assumed from prose alone. Check dimension 4 FIRST, ahead of 1-3, and mark it UNSTATED unless the parameter shape and return type/order are stated in so many words.
""".strip()

# The verdict must start its line (markdown decoration allowed), so "NO_QUESTION: <reason>" and
# mid-sentence mentions of the word are never mistaken for a question.
_QUESTION_RE = re.compile(r"^[ \t>*_`#-]*QUESTION[ \t]*:[ \t*_`]*(.+)$", re.IGNORECASE | re.MULTILINE)
_NO_QUESTION_RE = re.compile(r"\bNO_QUESTION\b", re.IGNORECASE)

# Used for the final generation, after the ask/no-ask decision.
CODE_PROMPT = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

The function must be named exactly `{entry_point}`.

IMPORTANT:
1. Before writing the code, you MUST write a step-by-step logic plan inside a Python docstring or comment. Discuss how you will handle edge cases (empty lists, negative numbers, ties) and the core algorithm.
2. Where an answer to a clarification question conflicts with the original requirement, the answer is authoritative.
3. You MUST include an `if __name__ == '__main__':` block at the end of the script containing your tests.

When writing the tests inside this block, you MUST follow these rules in order:
1. EXTRACT ALL EXAMPLES AND TESTS from the prompt above (like `>>>` doctests or assertions) and write them as strict `assert` statements. If the user provided a test case, it MUST be executed.
2. After the user's tests, write at least 3-5 of your own additional `assert` statements covering boundary cases (e.g., empty inputs, negatives, zeroes) and stress testing.

Enclose your entire solution in ```python and ```.
""".strip()


@dataclass
class _DoctestReport:
    """Result of mechanically extracting and running the requirement's own `>>>` examples."""

    ran: bool
    total: int
    passed: int
    failures: list[str] = field(default_factory=list)

    @staticmethod
    def empty() -> "_DoctestReport":
        return _DoctestReport(ran=False, total=0, passed=0, failures=[])


class StructuredClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "generation_attempts": 3,
        "max_questions": 1,
        "judge_attempts": 2,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        resolved: list[tuple[str, str]] = []

        try:
            for _ in range(self.config["max_questions"]):
                if not env.can_ask():
                    break
                spec = self._with_clarifications(prompt, resolved)
                seed = self._generate_seed(env, spec, entry_point)
                if seed is None:
                    break
                evidence = self._doctest_evidence(env, spec, seed)
                question = self._judge(env, spec, entry_point, seed, evidence)
                if question is None:
                    break
                answer = env.ask_human(question)
                resolved.append((question, answer))
        except (TooManyQuestionException, LimitsExceededException):
            pass

        spec = self._with_clarifications(prompt, resolved)
        return self._generate(env, spec, entry_point)

    # --- Seed + Judge ---

    def _generate_seed(self, env: ClarificationEnvironment, prompt: str, entry_point: str) -> str | None:
        messages = [
            {"role": "user", "content": SEED_CODE_PROMPT.format(entry_point=entry_point, prompt=prompt)}
        ]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
            try:
                return self._strip_main_block(_validate_and_parse_evalplus_result(response))
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(error)},
                ]
        return None

    def _doctest_evidence(self, env: ClarificationEnvironment, prompt: str, seed: str) -> str:
        asserts = self._extract_doctest_asserts(prompt)
        if not asserts:
            return ""
        report = self._check_doctests(env, seed, asserts)
        if not report.ran:
            return (
                f"Note: the candidate crashed before any of the {report.total} worked example(s) "
                "already in the requirement could be checked against it."
            )
        if report.passed == report.total:
            return ""
        lines = "\n".join(f"- {failure}" for failure in report.failures)
        return (
            f"Note: {report.total - report.passed} of {report.total} worked example(s) already in "
            f"the requirement FAIL against this candidate:\n{lines}"
        )

    @staticmethod
    def _extract_doctest_asserts(prompt: str) -> list[str]:
        """Best-effort, non-LLM extraction of the prompt's own `>>>` examples as assert lines.

        Never raises: a prompt with no doctests, or malformed ones, just yields an empty list.
        """
        try:
            examples = doctest.DocTestParser().get_examples(prompt)
        except Exception:
            return []

        asserts = []
        for example in examples:
            source = example.source.strip()
            want = example.want.strip()
            if not want:
                continue
            # DocTestParser doesn't understand Python syntax: when an example is the last
            # thing before a docstring closes, the closing triple-quote is swallowed into
            # `want`. Strip it, since almost every HumanEval-style prompt ends this way.
            for quote in ('"""', "'''"):
                if want.endswith(quote):
                    want = want[: -len(quote)].rstrip()
                    break
            if not want:
                continue
            try:
                ast.parse(source, mode="eval")
            except SyntaxError:
                continue
            try:
                expected = ast.literal_eval(want)
            except (ValueError, SyntaxError):
                continue
            asserts.append(f"assert ({source}) == {expected!r}")
        return asserts

    def _check_doctests(
        self, env: ClarificationEnvironment, seed_code: str, asserts: list[str]
    ) -> _DoctestReport:
        if not asserts:
            return _DoctestReport.empty()

        checks = "\n".join(
            f"try:\n    {a}\n    print('DOCTEST_{i}=OK')\nexcept Exception:\n    print('DOCTEST_{i}=FAIL')"
            for i, a in enumerate(asserts)
        )
        output = env.exec_code(f"{seed_code}\n{checks}")

        if not any(f"DOCTEST_{i}=" in output for i in range(len(asserts))):
            return _DoctestReport(ran=False, total=len(asserts), passed=0, failures=[])

        failures = [asserts[i] for i in range(len(asserts)) if f"DOCTEST_{i}=OK" not in output]
        return _DoctestReport(
            ran=True, total=len(asserts), passed=len(asserts) - len(failures), failures=failures
        )

    @staticmethod
    def _has_worked_example(prompt: str) -> bool:
        """Whether the requirement shows a `>>>` call at all, regardless of whether it's well-formed
        enough for `_extract_doctest_asserts` to turn into a checkable assert.

        HumanEval prompts almost always have one; Mbpp's one-liners almost never do. Purely
        mechanical (a substring check), so this costs nothing extra to compute.
        """
        return ">>>" in prompt

    @staticmethod
    def _has_typed_signature(prompt: str, entry_point: str) -> bool:
        """Whether the requirement itself opens with a `def {entry_point}(...)` line that already
        carries a type annotation - a parameter's `: type` or a `-> type` return annotation.

        HumanEval prompts are typed function stubs, so this is true even on variants that dropped
        their `>>>` example; Mbpp's one-liners never contain a `def` line at all, so this is always
        false for them. Purely mechanical (a regex match), no extra cost.
        """
        match = re.search(rf"def\s+{re.escape(entry_point)}\s*\(([^)]*)\)\s*(->\s*[^:]+)?:", prompt)
        if not match:
            return False
        params, return_annotation = match.group(1), match.group(2)
        return bool(return_annotation) or ":" in params

    def _judge(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        candidate: str,
        evidence: str,
    ) -> str | None:
        no_shape_signal = not (
            self._has_worked_example(prompt) or self._has_typed_signature(prompt, entry_point)
        )
        judge_prompt = JUDGE_PROMPT.format(
            entry_point=entry_point,
            prompt=prompt,
            candidate=candidate,
            evidence_block=f"\n{evidence}\n" if evidence else "",
            priority_override=f"\n{_NO_SHAPE_SIGNAL_PRIORITY_OVERRIDE}\n" if no_shape_signal else "",
        )
        # A formatting slip in the judge's reply must not silently become "don't ask": only an
        # explicit NO_QUESTION means no. Retry, then give up without a question.
        for _ in range(self.config["judge_attempts"]):
            understood, question = self._parse_judge_response(env.llm(judge_prompt))
            if understood:
                return question
        return None

    @staticmethod
    def _parse_judge_response(response: str) -> tuple[bool, str | None]:
        """Returns (understood, question). understood is False only if no verdict was found."""
        question_matches = list(_QUESTION_RE.finditer(response))
        no_question_matches = list(_NO_QUESTION_RE.finditer(response))
        last_question = question_matches[-1] if question_matches else None
        last_no_question = no_question_matches[-1] if no_question_matches else None

        # The final verdict wins: reasoning text may mention either token earlier on.
        if last_question and (not last_no_question or last_question.start() > last_no_question.start()):
            text = last_question.group(1).strip(" *_`")
            return (True, text) if text else (False, None)
        if last_no_question:
            return True, None
        return False, None

    # --- Final Generation ---

    def _generate(self, env: ClarificationEnvironment, spec: str, entry_point: str) -> str:
        messages = [
            {"role": "user", "content": CODE_PROMPT.format(entry_point=entry_point, prompt=spec)}
        ]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
            try:
                raw_code = _validate_and_parse_evalplus_result(response)
                test_code = f"{raw_code}\nprint('ALL_TESTS_PASSED')"
                test_result = env.exec_code(test_code)

                if "ALL_TESTS_PASSED" not in test_result:
                    raise ValueError(f"Execution failed or asserts crashed. Traceback/Output:\n{test_result}")

                return self._strip_main_block(raw_code)
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            f"The execution failed with the following traceback/output:\n{error}\n"
                            "First, identify the root cause of the error in 1 sentence. Then, provide "
                            "the complete, corrected Python script in a ```python block. Ensure all tests pass."
                        ),
                    },
                ]

        try:
            return self._strip_main_block(_validate_and_parse_evalplus_result(response))
        except ValueError:
            return response

    @staticmethod
    def _strip_main_block(code: str) -> str:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return code
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) in {"__name__ == '__main__'", "'__main__' == __name__"}
        ]
        if not guards:
            return code
        lines = code.split("\n")
        for node in reversed(guards):
            del lines[node.lineno - 1 : node.end_lineno]
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _with_clarifications(prompt: str, resolved: list[tuple[str, str]]) -> str:
        fragments = [
            f"Question #{i}:\n{question}\nAnswer:\n{answer}\n"
            for i, (question, answer) in enumerate(resolved, start=1)
        ]
        return "\n".join([prompt, *fragments])
