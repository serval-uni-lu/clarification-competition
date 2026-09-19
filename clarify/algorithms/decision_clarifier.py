# SPDX-FileCopyrightText: 2026 Yongcheng Huang <Y.Huang-12@tudelft.nl>
#
# SPDX-License-Identifier: MIT

"""DecisionClarifier: clarify the most consequential unresolved decision.

A draft exposes behavioral assumptions. An independent critic compares plausible
interpretations using a distinguishing example, selects one atomic requirement,
and estimates whether asking can improve correctness enough to justify its cost.
The implementation is then reviewed against the prompt and any actual answer.
Optional checks use only explicit requirements and run through the SDK sandbox.

Team: D4vidHuang
Team Members: Yongcheng Huang
Main Contact: Y.Huang-12@tudelft.nl
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment


_SUCCESS_MARKER = "DECISION_CLARIFIER_CHECKS_OK_7C82B09E"


def _json_object(response: str | None) -> dict[str, Any]:
    """Accept ordinary JSON, a JSON fence, or a short surrounding explanation."""
    if not isinstance(response, str):
        return {}
    decoder = json.JSONDecoder()
    # Parsing never executes model output. Failed formatting spends no extra call.
    for match in list(re.finditer(r"\{", response))[:32]:
        try:
            value, _ = decoder.raw_decode(response[match.start() :])
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _valid_code(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
        # Compilation checks semantic syntax (e.g. a top-level return) without
        # executing either the implementation or any import in model output.
        compile(tree, "<candidate>", "exec")
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == entry_point
        for node in tree.body
    )


def _extract_code(response: str | None, entry_point: str) -> str:
    if not isinstance(response, str):
        return ""
    candidates = []
    structured = _json_object(response).get("code")
    if isinstance(structured, str):
        candidates.append(structured)
        candidates.extend(re.findall(r"```(?:python|py)?\s*\n(.*?)```", structured, re.S))
    candidates.extend(re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S))
    candidates.append(response)
    for candidate in candidates:
        candidate = candidate.strip()
        if _valid_code(candidate, entry_point):
            return candidate
    return ""


def _question(decision: dict[str, Any], threshold: float, max_chars: int) -> str:
    """Require an explicit, checkable decision record before spending a turn."""
    if (
        decision.get("ask") is not True
        or decision.get("atomic") is not True
        or decision.get("answer_in_prompt") is not False
    ):
        return ""
    gain = decision.get("expected_gain")
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not threshold <= gain <= 1.0
    ):
        return ""
    alternatives = decision.get("alternatives")
    witness = decision.get("witness")
    question = decision.get("question")
    if (
        not isinstance(alternatives, list)
        or not 2 <= len(alternatives) <= 4
        or any(not isinstance(item, str) or not item.strip() for item in alternatives)
        or len(set(alternatives)) != len(alternatives)
        or not isinstance(witness, str)
        or not witness.strip()
        or not isinstance(question, str)
    ):
        return ""
    question = question.strip()
    if not question or len(question) > max_chars or "\n" in question:
        return ""
    if question.count("?") != 1 or not question.endswith("?"):
        return ""
    return question


class DecisionClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "max_llm_calls": 4,  # Draft, critic, final review, and at most one repair.
        "max_questions": 1,  # Either zero (ablation) or one atomic clarification.
        "use_critic": True,  # Independently review decisions; false uses the draft's ranking.
        "ambiguity_limit": 4,  # Maximum competing behavioral decisions in the draft.
        "min_expected_gain": 0.05,  # Minimum estimated absolute correctness gain.
        "max_question_chars": 360,  # Keep the single fact request compact.
        "sandbox_checks": True,  # Run generated checks via env.exec_code, if present.
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, str]) -> str:
        # These are the only problem fields read by the submission.
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        if not isinstance(entry_point, str) or not entry_point.isidentifier() or keyword.iskeyword(entry_point):
            raise ValueError("entry_point must be a Python function name")
        max_calls = self.config["max_llm_calls"]
        max_questions = self.config["max_questions"]
        ambiguity_limit = self.config["ambiguity_limit"]
        threshold = self.config["min_expected_gain"]
        max_chars = self.config["max_question_chars"]
        if type(max_calls) is not int or not 1 <= max_calls <= 4:
            raise ValueError("max_llm_calls must be an integer from 1 to 4")
        if type(max_questions) is not int or max_questions not in (0, 1):
            raise ValueError("max_questions must be 0 or 1")
        if type(ambiguity_limit) is not int or not 1 <= ambiguity_limit <= 8:
            raise ValueError("ambiguity_limit must be an integer from 1 to 8")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("min_expected_gain must be between 0 and 1")
        if type(max_chars) is not int or max_chars < 32:
            raise ValueError("max_question_chars must be an integer of at least 32")
        for option in ("sandbox_checks", "use_critic"):
            if type(self.config[option]) is not bool:
                raise ValueError(f"{option} must be a boolean")

        calls = 0
        unavailable = False

        def model_available() -> bool:
            nonlocal unavailable
            if unavailable or calls >= max_calls:
                return False
            try:
                if env.prompt_cost >= env.prompt_budget:
                    unavailable = True
            except Exception:
                unavailable = True
            return not unavailable

        def call(system: str, payload: dict[str, Any]) -> str | None:
            nonlocal calls, unavailable
            if not model_available():
                return None
            calls += 1
            try:
                return env.llm([
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ])
            except Exception:
                # Includes SDK budget exhaustion and provider failures. Preserve
                # the last structurally valid program and make no further calls.
                unavailable = True
                return None

        task = {"prompt": prompt, "entry_point": entry_point}
        draft = call(
            "You implement Python functions from user requirements. Read the supplied task as "
            "the specification, including any signature or examples. Produce a complete, "
            "self-contained implementation with the exact entry point. Before settling on "
            "behavior, identify genuinely unresolved choices with different observable outputs. "
            "Do not invent uncertainty about requirements already stated or mathematically "
            "implied; do not rely on remembering a benchmark or its solution. Distinguish "
            "semantic gaps from ordinary implementation bugs. Return one JSON object: "
            '{"code":"complete Python source", "ambiguities":[{"decision":"one missing fact",'
            '"alternatives":["interpretation A","interpretation B"],'
            '"witness":"a small invented input and how the outputs differ",'
            '"assumption_in_code":"the choice the draft makes",'
            '"question":"one atomic fact question ending in ?",'
            '"ask":true,"atomic":true,"answer_in_prompt":false,'
            '"expected_gain":0.0}]}. '
            f"Include at most {ambiguity_limit} ambiguities, highest impact first; an empty list is valid. "
            "For each, estimate the absolute correctness probability gain from its answer "
            "(0.10 means ten percentage points). Ask exactly one fact per question; never "
            "request code, an algorithm, tests, all requirements, or a correctness guarantee. "
            "Choose reasonable conventional assumptions in the executable draft. Do not ask "
            "the user yet. JSON string newlines must be escaped.",
            task,
        )
        best_code = _extract_code(draft, entry_point)
        draft_object = _json_object(draft)
        hypotheses = draft_object.get("ambiguities", [])
        if not isinstance(hypotheses, list):
            hypotheses = []
        hypotheses = hypotheses[:ambiguity_limit]
        decision: dict[str, Any] = {}
        clarification: dict[str, str] | None = None

        can_ask = False
        reserved_calls = 1 + int(self.config["use_critic"])
        if max_questions and model_available() and calls + reserved_calls <= max_calls:
            try:
                can_ask = env.can_ask()
            except Exception:
                pass
        if can_ask and self.config["use_critic"]:
            critique = call(
                "Independently audit this programming specification and provisional solution. "
                "The draft and its hypotheses may be wrong. Find the single unresolved fact "
                "whose answer is most likely to change the correct implementation. Consider "
                "missing interface details, ordering, boundaries, conventions and defaults "
                "only where two plausible interpretations cause different behavior on a "
                "concrete input. Eliminate issues answered by the prompt, examples, standard "
                "mathematics or earlier answers, and issues solvable by fixing a coding bug. "
                "Construct your own distinguishing witness; never ask for tests, reference "
                "code, an algorithm, a list of all requirements, or a correctness guarantee. "
                "The question must ask exactly ONE fact, with a precise objective answer. "
                "Two alternative values for that same fact are allowed; multiple independent "
                "decisions are not. A short question about one important omission can be "
                "worth asking even if the draft made a conventional guess. Estimate the "
                "absolute probability gain in final implementation correctness from this "
                "answer: 0.10 means ten percentage points, not confidence that an ambiguity "
                "exists. Asking costs roughly four percent of a successful answer, so choose "
                "the highest-value fact, or abstain when the likely benefit is smaller. "
                "Return only JSON: "
                '{"ask":true or false,"question":"one concise question ending in ?",'
                '"atomic":true or false,"answer_in_prompt":true or false,'
                '"expected_gain":0.0,"alternatives":["value A","value B"],'
                '"witness":"invented input; predicted output under each value",'
                '"reason":"why this is unresolved and consequential",'
                '"code_review":"specific implementation defects, or empty string"}. '
                "Do not answer your own clarification question or assume a hidden fact.",
                {**task, "draft": best_code or draft or "", "hypotheses": hypotheses},
            )
            decision = _json_object(critique)
        elif can_ask:
            eligible = [
                hypothesis for hypothesis in hypotheses
                if isinstance(hypothesis, dict) and _question(hypothesis, threshold, max_chars)
            ]
            if eligible:
                decision = max(eligible, key=lambda item: item["expected_gain"])
        if can_ask:
            question = _question(decision, threshold, max_chars)
            # The final synthesis slot was reserved before spending a human turn.
            if question and model_available():
                try:
                    if env.can_ask():
                        answer = env.ask_human(question)
                        if isinstance(answer, str) and answer.strip():
                            clarification = {"question": question, "answer": answer}
                except Exception:
                    # A changed question budget must not discard the usable draft.
                    pass

        final = call(
            "Review and finish the Python implementation using the task and any actual "
            "clarification answer. Preserve the required function name, signature and "
            "return shape. Resolve implementation defects, trace representative edge cases, "
            "and check imports, termination, mutation, indexing and numerical behavior. "
            "The critic is a fallible review, not additional requirements. Unanswered "
            "hypotheses are NOT facts. A vague or refused answer does not establish a "
            "specific choice; use the original prompt and justified conventional defaults. "
            "Provide the simplest correct self-contained function and helpers, without "
            "top-level example execution. Optionally provide a few small assert statements "
            "whose expected results follow directly from explicit prompt examples, the "
            "actual answer, or unambiguous mathematical properties. Do not turn unresolved "
            "assumptions or a draft's output into test expectations. Use an empty checks "
            "string if there are no justified checks. Return only JSON "
            '{"code":"complete Python source","checks":"optional Python asserts"}. '
            "Escape newlines in JSON strings.",
            {
                **task,
                "draft": best_code or draft or "",
                "review": decision,
                "clarification": clarification,
            },
        )
        final_code = _extract_code(final, entry_point)
        if final_code:
            best_code = final_code
        failure = ""
        if final is not None and not final_code:
            failure = "The final response was not valid Python with the required top-level function."
        checks = _json_object(final).get("checks", "")
        if (
            final_code and self.config["sandbox_checks"] and isinstance(checks, str)
            and checks.strip() and model_available()
        ):
            try:
                ast.parse(checks)
            except (SyntaxError, ValueError, RecursionError):
                checks = ""
            if checks.strip():
                try:
                    output = env.exec_code(
                        final_code + "\n\n" + checks + f'\nprint("{_SUCCESS_MARKER}")\n'
                    )
                    if not any(line.strip() == _SUCCESS_MARKER for line in str(output).splitlines()):
                        failure = "Self-generated checks failed. Sandbox output:\n" + str(output)
                except Exception:
                    # Sandbox infrastructure errors are not evidence of a code bug.
                    pass
        if failure:
            repaired = call(
                "Repair this Python solution. Use the task and actual clarification as "
                "requirements. Check whether a failing self-generated assertion is itself "
                "justified; discard incorrect assertions instead of changing correct "
                "behavior to satisfy them. Produce the full self-contained implementation "
                "with the exact required function name. Return only a fenced python block.",
                {
                    **task,
                    "clarification": clarification,
                    "candidate": final_code or best_code or final or draft or "",
                    "checks": checks,
                    "failure": failure,
                },
            )
            repaired_code = _extract_code(repaired, entry_point)
            if repaired_code:
                best_code = repaired_code
        if not best_code:
            # Explicit failure is preferable to returning malformed source. This
            # is a last-resort artifact, never claimed to solve the coding task.
            best_code = (
                f"def {entry_point}(*args, **kwargs):\n"
                '    raise RuntimeError("No valid implementation was generated within the budget.")'
            )
        # The released evaluator requires the fence even though the public API
        # describes the return value as either a script or a fenced script.
        return "```python\n" + best_code + "\n```"
