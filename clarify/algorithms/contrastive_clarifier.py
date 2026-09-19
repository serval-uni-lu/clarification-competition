# SPDX-FileCopyrightText: 2026 Dinh <dinhtd.b23cc038@stu.ptit.edu.vn>
# SPDX-License-Identifier: MIT

"""ContrastiveClarifier: select one behavior-changing requirement to clarify.

Team: Dinh
Team Members: Dinh
Main Contact: dinhtd.b23cc038@stu.ptit.edu.vn
"""

import ast
import json
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import TooManyQuestionException

ANALYZE = """Analyze the public programming requirement, not any hidden benchmark.
Do not write code yet. Identify at most {limit} independent unresolved behavioral
choices. Do not invent ambiguity when the prompt, examples, or conventional meaning
already settle it. For each candidate give two plausible interpretations and one
small input on which their outputs differ. These are YOUR hypothetical examples,
not questions asking the user for test cases. Reject stylistic/algorithm choices.
Rank candidates by likely impact on correct behavior. Ask about one fact only.
Return JSON: {{"candidates": [{{"issue": "...", "interpretation_a": "...",
"interpretation_b": "...", "witness_input": "...", "output_a": "...",
"output_b": "...", "question": "one atomic question"}}]}}.
Return an empty list when nothing genuinely needs clarification.
"""

REVIEW = """Independently review these proposed ambiguities against the original
requirement. Proposals are fallible, not additional requirements. Select at most
one question: it must resolve ONE behavior-changing fact, genuinely unspecified
or contradictory, not inferable from the prompt/examples, not an implementation
choice, and not a request for tests/reference code. Check the two interpretations
are plausible and their witness really distinguishes them. Reject speculative
edge cases and questions already answered. A single sentence can still bundle
multiple facts: reject it. Do not ask the user to provide the witness output.
Return JSON {"ask": true, "question": "..."} or {"ask": false}.
"""

GENERATE = """Implement the requested Python function. Use only the standard
library. Preserve the required function name/signature. Public requirements and
the user's clarification are authoritative. If clarification is inconclusive,
use the most defensible reading of the original prompt; do not invent facts.
Return only a fenced python implementation, no questions or explanations.
"""


def parse_object(text):
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def extract_code(response, entry_point):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    for code in blocks + [response]:
        try:
            tree = ast.parse(code.strip())
        except (SyntaxError, ValueError):
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
            for node in tree.body
        ):
            return code.strip()
    return None


class ContrastiveClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "max_candidates": 3,  # Prompt cap on proposed behavioral ambiguities.
        "review_question": True,  # Disable only for a predeclared ablation.
        "repair_attempts": 1,  # Syntax/entry-point repairs, not hidden-test feedback.
        "generation_reserve_fraction": 0.5,  # Stop planning after half the cost budget.
    }

    def validate_config(self):
        if not 1 <= int(self.config["max_candidates"]) <= 5:
            raise ValueError("max_candidates must be between 1 and 5")
        if not 0 <= int(self.config["repair_attempts"]) <= 2:
            raise ValueError("repair_attempts must be between 0 and 2")
        if not 0 < float(self.config["generation_reserve_fraction"]) < 1:
            raise ValueError("generation_reserve_fraction must be between 0 and 1")
        if not isinstance(self.config["review_question"], bool):
            raise ValueError("review_question must be a boolean")

    def run(self, env, problem):
        self.validate_config()
        # Deliberately read only public prompt/entry point, never task IDs or oracles.
        public = json.dumps(
            {"prompt": problem["prompt"], "entry_point": problem["entry_point"]}, ensure_ascii=False
        )
        question = ""
        clarification = None
        planning_limit = env.prompt_budget * (1 - float(self.config["generation_reserve_fraction"]))
        if env.can_ask() and env.prompt_cost < planning_limit:
            analysis = parse_object(
                env.llm(
                    [
                        {
                            "role": "system",
                            "content": ANALYZE.format(limit=self.config["max_candidates"]),
                        },
                        {"role": "user", "content": public},
                    ]
                )
            )
            candidates = analysis.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
            required = (
                "issue",
                "interpretation_a",
                "interpretation_b",
                "witness_input",
                "output_a",
                "output_b",
                "question",
            )
            candidates = [
                c
                for c in candidates
                if isinstance(c, dict)
                and all(isinstance(c.get(k), str) and c[k].strip() for k in required)
                and c["output_a"].strip() != c["output_b"].strip()
            ][: int(self.config["max_candidates"])]
            if candidates:
                if self.config["review_question"]:
                    if env.prompt_cost < planning_limit:
                        decision = parse_object(
                            env.llm(
                                [
                                    {"role": "system", "content": REVIEW},
                                    {
                                        "role": "user",
                                        "content": public
                                        + "\nProposals:\n"
                                        + json.dumps(candidates),
                                    },
                                ]
                            )
                        )
                        if decision.get("ask") is True and isinstance(
                            decision.get("question"), str
                        ):
                            question = decision["question"].strip()
                else:
                    question = candidates[0]["question"].strip()
        # Cheap format guard; semantic atomicity is reviewed by the model, not proven.
        if (
            question
            and len(question) <= 500
            and question.count("?") <= 1
            and "\n" not in question
            and "```" not in question
            and env.can_ask()
        ):
            try:
                clarification = env.ask_human(question)
            except TooManyQuestionException:
                clarification = None
        messages = [{"role": "system", "content": GENERATE}, {"role": "user", "content": public}]
        if clarification is not None:
            messages.extend(
                [
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": clarification},
                ]
            )
        # Do not feed speculative interpretations into final generation.
        response = env.llm(messages)
        for attempt in range(int(self.config["repair_attempts"]) + 1):
            code = extract_code(response, problem["entry_point"])
            if code is not None:
                return "```python\n" + code + "\n```"
            if (
                attempt == int(self.config["repair_attempts"])
                or env.prompt_cost >= env.prompt_budget
            ):
                break
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": "Return syntactically valid Python defining "
                        + problem["entry_point"]
                        + ". Return code only, not a clarification question.",
                    },
                ]
            )
            response = env.llm(messages)
        return response
