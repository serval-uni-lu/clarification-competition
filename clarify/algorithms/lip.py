# SPDX-FileCopyrightText: 2026 Yang Zhang <zhangy2256@mail2.sysu.edu.cn>
#
# SPDX-License-Identifier: MIT

"""IntentBridgeClarifier.

Propose a common program for plausible interpretations of the visible task and
audit its compatibility. On conflicting observable outputs, ask one atomic
behavioural question and synthesize afresh using the actual answer. Compatibility
audits are fallible model judgments, not access to hidden tests or formal proofs.

Team: lip
Team Members: Yang Zhang (Sun Yat-sen University), Zhen Wang (Sun Yat-sen University), Wenjie Hu (Sun Yat-sen University)
Main Contact: Yang Zhang <zhangy2256@mail2.sysu.edu.cn>
"""

from __future__ import annotations

import ast
import json
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import LimitsExceededException, TooManyQuestionException


def json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected object")
    return value


def python_code(text, entry):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = match.group(1) if match else text
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
        raise ValueError("Wrong entry point")
    return code


class _TaskRun:
    def record(self, **values):
        self.trace.append(values)

    def ask(self, env, question):
        if not isinstance(question, str) or not question.strip() or not env.can_ask():
            return ""
        try:
            return env.ask_human(question)
        except TooManyQuestionException:
            return ""

    def gate(self, env, problem):
        self.gate_attempted = True
        if not env.can_ask():
            return ""
        response = env.llm(
            "Review this Python task. Ask at most ONE atomic behavioral fact whose unknown value could cause failure. "
            'Do not ask for algorithms, test cases, style, or already stated facts. Return only JSON {"question":"..."}; '
            "use an empty string if clear.\n" + json.dumps(problem)
        )
        return json_object(response).get("question", "")

    def fallback(self, env, problem):
        if not self.qa[0] and not self.gate_attempted and env.can_ask():
            try:
                question = self.gate(env, problem)
                self.qa = (question, self.ask(env, question))
            except (ValueError, KeyError, TypeError, LimitsExceededException):
                pass
        return self.solve(env, problem, *self.qa)

    def solve(self, env, problem, question="", answer="", note=""):
        question = question if isinstance(question, str) else ""
        answer = answer if isinstance(answer, str) else ""
        prompt = (
            "Implement exactly the entry_point in this task, even if names elsewhere differ. Return a complete fenced Python function.\n"
            + json.dumps(problem)
        )
        if question:
            prompt += (
                "\nActual clarification question: " + question + "\nActual user answer: " + answer
            )
            prompt += "\nThe actual answer supersedes only contradicted requirements; retain other explicit constraints."
        if note:
            prompt += "\nAuditable working state (not extra user requirements):\n" + note
        messages = [{"role": "user", "content": prompt}]
        for _attempt in range(self.config["repair_attempts"] + 1):
            text = env.llm(messages)
            try:
                return python_code(text, problem["entry_point"])
            except (SyntaxError, ValueError):
                messages += [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "Return complete valid Python implementing exactly "
                        + problem["entry_point"]
                        + " in a python fence.",
                    },
                ]
        return text

    def robust_synthesis(self, env, problem):
        proposal = json_object(
            env.llm(
                "Consider at most 3 plausible behavioral interpretations of the visible task. "
                "Try to construct ONE whole program compatible with all, not a sampled program vote. If output requirements conflict, "
                'give ONE atomic question resolving the conflict and a concrete witness. JSON {"worlds":["..."],"common_code":"complete Python or empty",'
                '"compatibility_argument":"...","witness":null,"question":"..."}. Witness if conflicting: '
                '{"input":"example input described as data","required_outputs":["output under interpretation 1","output under interpretation 2"]}. '
                "No found witness does NOT prove compatibility. Ask no algorithms or hidden tests.\n"
                + json.dumps(problem)
            )
        )
        common = proposal.get("common_code", "")
        if common and proposal.get("witness") is None:
            code = python_code(common, problem["entry_point"])
            self.last_code = code
            audit = json_object(
                env.llm(
                    "Audit the proposed common implementation against each STATED interpretation. "
                    "This is a fallible compatibility check, not hidden-test validation. If any incompatible observable outputs exist, "
                    'return an atomic behavioral question. JSON {"compatible":true,"question":"","reason":"..."}. '
                    "Do not declare compatible merely because no tests failed.\n"
                    + json.dumps({"task": problem, "proposal": proposal})
                )
            )
            if audit.get("compatible") is True and not audit.get("question"):
                self.record(
                    status="applied",
                    route="proxy_common_program",
                    formal_compatibility_proved=False,
                )
                return code
            self.last_code = ""
            question = audit.get("question", "")
        else:
            question = proposal.get("question", "")
        answer = self.ask(env, question)
        self.qa = (question, answer)
        self.record(status="applied", route="conflict_question", formal_compatibility_proved=False)
        return self.solve(env, problem, question, answer)

    def execute(self, env, problem):
        self.trace = []
        self.qa = ("", "")
        self.gate_attempted = False
        self.last_code = ""
        try:
            return self.robust_synthesis(env, problem)
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            SyntaxError,
            LimitsExceededException,
        ):
            try:
                return self.fallback(env, problem)
            except LimitsExceededException:
                return self.last_code or (
                    "def "
                    + problem["entry_point"]
                    + "(*args, **kwargs):\n    raise NotImplementedError"
                )


class IntentBridgeClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        # Extra attempts to obtain syntactically valid target-function code.
        "repair_attempts": 1,
    }

    def run(self, env, problem) -> str:
        # The harness may reuse the algorithm object across tasks.
        state = _TaskRun()
        state.config = dict(self.config)
        visible = {"prompt": problem["prompt"], "entry_point": problem["entry_point"]}
        return state.execute(env, visible)
