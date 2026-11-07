# SPDX-FileCopyrightText: 2026 Shane Califano <scalifano4740@floridapoly.edu>
#
# SPDX-License-Identifier: MIT

"""ReqClarify

A clarification algorithm.

Team: NEXUS Lab
Team Members: Shane Califano, Mohamed Sylla, William Blaine, Karim Elish
Main Contact: kelish@floridapoly.edu
"""

from __future__ import annotations

import ast
import re
from typing import Any

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment

# --- prompts ---

SystemClarificationPrompt = """
You are a software requirements analyst.
Do NOT generate code.

Analyze the programming requirement. Ask a clarification question ONLY if there is a critical behavioral ambiguity that:
- Cannot be resolved by the function name, the provided examples, or standard Python conventions, AND
- Would produce different observable results on reasonable inputs, AND
- Is not a standard edge case (empty input, negative numbers, type differences, ordering of equal elements, etc.).

Rate the need for clarification:
0 = sufficiently specified or resolvable by standard conventions / function name / examples
1 = minor ambiguity safely resolved by conventions
2 = plausible ambiguity that might affect some cases
3 = critical missing information or contradiction that will cause incorrect behavior under normal inputs

Generate at most ONE question.
The question MUST be:
- Atomic (exactly one issue)
- Objective
- Extremely concise (≤ 12 words)
- Free of implementation or testing details

If need is 0 or 1, return QUESTION: NONE.

Return strictly in this format (no markdown, no extra text):
NEED: <0|1|2|3>
ISSUE: <short description or NONE>
INTERPRETATION_A: <behavior A or NONE>
INTERPRETATION_B: <behavior B or NONE>
QUESTION: <question or NONE>
"""

SystemCodePrompt = """
You are an expert Python programmer.

Implement the requested function according to the original requirement and, when provided, the user's clarification.
The user's clarification is authoritative for the issue that was asked about.
Do not invent unrelated requirements.

CRITICAL RULES (violations will cause test failures):
1. The function MUST be named exactly as specified in FUNCTION.
2. Every code path must have an explicit return value. NEVER write a bare `return` with nothing after it.
3. When comparing or indexing list/array elements, you MUST use proper indices. 
   - WRONG: `if arr < arr:` or `if arr > arr:`
   - CORRECT: `if arr[i] < arr[i-1]:` or `if arr[i] > arr[i+gap]:`
4. When checking for a contiguous sub-sequence, use slicing: `main[i:i+len(sub)] == sub`.
5. Match the parameter count and order implied by the requirement and examples. Prefer the minimal signature that matches how the function is likely to be called.
6. Use only ASCII characters.
7. The code must be complete and syntactically valid Python.

Return ONLY the Python implementation. No explanations, no tests, no markdown fences.
"""

SystemAuditPrompt = """
You are a strict software correctness auditor.

Check whether the candidate implementation satisfies the requirement and clarification (if any).

CRITICAL BUGS TO FIX (these appear frequently and must be repaired):
- Bare `return` with no value. Every path must return a concrete value.
- Missing indices in comparisons: `if arr < arr:` or `if arr > arr:` must become proper indexed comparisons such as `if arr[i] < arr[i-1]:` or `if arr[i] > arr[i+gap]:`.
- Contiguous sublist checks that compare entire lists instead of slices. Use `main[i:i+len(sub)] == sub`.
- Incorrect function arity relative to the examples or clarification.
- Incomplete logic that ignores the clarification answer.

Rules:
- Do NOT invent new requirements.
- Do NOT remove necessary loops or logic.
- If the code has any of the above bugs, repair them.
- If the implementation is already correct, leave it unchanged.
- Return ONLY the final Python implementation (ASCII only). No explanations.
"""


class ReqClarifyClarification(ClarificationAlgorithmBase):
    # we only ask when the model thinks the ambiguity is at least a 2
    DEFAULT_CONFIG = {
        "ask_threshold": 2,
        "max_questions": 1,
        "max_audit_attempts": 3,
    }

    def extract_code(self, raw_text: str) -> str:
        # grab the python block if the model wrapped it in fences
        if not raw_text:
            return ""
        match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
        else:
            code = raw_text.replace("```python", "").replace("```", "").strip()
        code = code.encode("ascii", "ignore").decode("ascii")
        lines = [line for line in code.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()

    def has_bare_return(self, code: str) -> bool:
        # true when a line is just "return" with nothing after it
        return bool(re.search(r"^\s*return\s*$", code, re.MULTILINE))

    def has_broken_index_pattern(self, code: str) -> bool:
        # catches the usual "if arr < arr:" mistake
        return bool(re.search(r"\bif\s+\w+\s*[<>]=?\s*\w+\s*:", code))

    def is_valid_python(self, code: str, entry_point: str = "") -> bool:
        # quick syntax check, plus make sure the expected function is defined
        if not code or len(code.strip()) < 10:
            return False
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        if entry_point:
            defined_funcs = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if entry_point not in defined_funcs:
                return False

        if self.has_bare_return(code):
            return False
        return True

    def validate_or_fallback(
        self, audited_code: str, candidate_code: str, entry_point: str = ""
    ) -> str:
        # use the audited version if it looks okay, otherwise keep the original
        if self.is_valid_python(audited_code, entry_point):
            return audited_code
        if self.is_valid_python(candidate_code, entry_point):
            return candidate_code
        return audited_code or candidate_code

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        cfg = getattr(self, "config", None) or self.DEFAULT_CONFIG
        ask_threshold = int(cfg.get("ask_threshold", 2))
        max_questions = int(cfg.get("max_questions", 1))
        max_audit_attempts = int(cfg.get("max_audit_attempts", 3))

        user_prompt = problem["prompt"]
        entry_point = problem.get("entry_point", "")

        # first, look at the requirement and decide if we should ask anything
        analysis_input = f"FUNCTION: {entry_point}\nREQUIREMENT:\n{user_prompt}"
        clarification_messages = [
            {"role": "system", "content": SystemClarificationPrompt},
            {"role": "user", "content": analysis_input},
        ]

        questions_asked = 0
        analysis = {
            "need": 0,
            "ask": False,
            "issue": "NONE",
            "interpretation_a": "NONE",
            "interpretation_b": "NONE",
            "question": "NONE",
        }
        answer: str | None = None

        while questions_asked < max_questions and env.can_ask():
            raw_analysis = env.llm(clarification_messages)
            clean_analysis = raw_analysis.replace("```text", "").replace("```", "").strip()
            clarification_messages.append({"role": "assistant", "content": clean_analysis})

            analysis = self.parse_clarification(clean_analysis, ask_threshold=ask_threshold)

            if (
                analysis["ask"]
                and analysis["question"]
                and analysis["question"].upper() != "NONE"
                and env.can_ask()
            ):
                questions_asked += 1
                answer = env.ask_human(analysis["question"])
                if answer:
                    answer = answer.strip()
                clarification_messages.append({"role": "user", "content": answer or ""})
            else:
                break

        # now write the function, using the clarification if we got one
        if answer is not None and answer.strip():
            code_user_input = (
                f"ORIGINAL REQUIREMENT:\n{user_prompt}\n\n"
                f"FUNCTION:\n{entry_point}\n\n"
                f"AMBIGUITY:\n{analysis['issue']}\n\n"
                f"INTERPRETATION A:\n{analysis['interpretation_a']}\n\n"
                f"INTERPRETATION B:\n{analysis['interpretation_b']}\n\n"
                f"CLARIFICATION QUESTION:\n{analysis['question']}\n\n"
                f"USER ANSWER:\n{answer}\n\n"
                f"Implement the function `{entry_point}`."
            )
        else:
            code_user_input = (
                f"ORIGINAL REQUIREMENT:\n{user_prompt}\n\n"
                f"FUNCTION:\n{entry_point}\n\n"
                f"Implement the function `{entry_point}`."
            )

        programming_messages = [
            {"role": "system", "content": SystemCodePrompt},
            {"role": "user", "content": code_user_input},
        ]
        candidate_raw = env.llm(programming_messages)
        candidate_code = self.extract_code(candidate_raw)

        # check the candidate against the requirement and the clarification
        audit_user_input = (
            f"ORIGINAL REQUIREMENT:\n{user_prompt}\n\n"
            f"FUNCTION:\n{entry_point}\n\n"
            f"CLARIFICATION QUESTION:\n{analysis['question'] if answer else 'NONE'}\n\n"
            f"USER ANSWER:\n{answer if answer else 'NONE'}\n\n"
            f"CANDIDATE CODE:\n```python\n{candidate_code}\n```"
        )
        audit_messages = [
            {"role": "system", "content": SystemAuditPrompt},
            {"role": "user", "content": audit_user_input},
        ]
        audited_raw = env.llm(audit_messages)
        audited_code = self.extract_code(audited_raw)

        code = self.validate_or_fallback(audited_code, candidate_code, entry_point)

        # if we can run the code, try to fix a few common problems
        if hasattr(env, "exec_code"):
            for _attempt in range(max_audit_attempts):
                clean_code = self.extract_code(code)
                if not clean_code:
                    break

                needs_fix = (
                    self.has_bare_return(clean_code)
                    or self.has_broken_index_pattern(clean_code)
                    or not self.is_valid_python(clean_code, entry_point)
                )

                try:
                    exec_result = str(env.exec_code(clean_code))
                except Exception as e:
                    exec_result = f"Exception: {e}"
                    needs_fix = True

                failed = needs_fix or any(
                    err in exec_result
                    for err in (
                        "Traceback",
                        "SyntaxError",
                        "IndentationError",
                        "NameError",
                        "TypeError",
                        "AssertionError",
                        "Exception:",
                    )
                )

                if failed:
                    error_hint = ""
                    if self.has_bare_return(clean_code):
                        error_hint += "The code contains bare `return` statements with no value. Every path must return a concrete value. "
                    if self.has_broken_index_pattern(clean_code):
                        error_hint += "The code has comparisons like `if arr < arr:` or `if arr > arr:` that are missing indices. Fix them to use proper indexing such as `arr[i] < arr[i-1]` or `arr[i] > arr[i+gap]`. "
                    if "main == sub" in clean_code or "mainlist == sublist" in clean_code:
                        error_hint += "Contiguous sublist checks must use slicing: `main[i:i+len(sub)] == sub`. "

                    error_message = (
                        f"The code has bugs and failed validation.\n"
                        f"Error / result:\n{exec_result}\n\n"
                        f"{error_hint}\n"
                        f"Fix all issues so the function is complete, returns values on every path, "
                        f"uses correct indexing, matches the expected signature, and satisfies the requirement + clarification. "
                        f"Return ONLY the corrected Python code."
                    )
                    audit_messages.append({"role": "user", "content": error_message})
                    repaired_raw = env.llm(audit_messages)
                    repaired_code = self.extract_code(repaired_raw)
                    code = self.validate_or_fallback(repaired_code, clean_code, entry_point)
                    audit_messages.append({"role": "assistant", "content": code})
                else:
                    code = clean_code
                    break

        return self.extract_code(code)

    def parse_clarification(self, clarification: str, ask_threshold: int = 2) -> dict[str, Any]:
        # pull the fields out of whatever the model returned
        outputs: dict[str, Any] = {
            "need": 0,
            "ask": False,
            "issue": "NONE",
            "interpretation_a": "NONE",
            "interpretation_b": "NONE",
            "question": "NONE",
        }

        if not clarification:
            return outputs

        need_match = re.search(r"NEED:\s*(\d+)", clarification, re.IGNORECASE)
        if need_match:
            need_val = int(need_match.group(1))
            outputs["need"] = need_val
            outputs["ask"] = need_val >= ask_threshold

        issue_match = re.search(
            r"ISSUE:\s*(.*?)(?=INTERPRETATION_A:|$)", clarification, re.IGNORECASE | re.DOTALL
        )
        if issue_match:
            outputs["issue"] = issue_match.group(1).strip()

        int_a_match = re.search(
            r"INTERPRETATION_A:\s*(.*?)(?=INTERPRETATION_B:|$)",
            clarification,
            re.IGNORECASE | re.DOTALL,
        )
        if int_a_match:
            outputs["interpretation_a"] = int_a_match.group(1).strip()

        int_b_match = re.search(
            r"INTERPRETATION_B:\s*(.*?)(?=QUESTION:|$)", clarification, re.IGNORECASE | re.DOTALL
        )
        if int_b_match:
            outputs["interpretation_b"] = int_b_match.group(1).strip()

        question_match = re.search(r"QUESTION:\s*(.*)", clarification, re.IGNORECASE | re.DOTALL)
        if question_match:
            question_text = question_match.group(1).strip()
            first_line = question_text.splitlines()[0].strip() if question_text else "NONE"
            if first_line.upper() == "NONE" or not first_line:
                outputs["question"] = "NONE"
            else:
                words = first_line.split()
                if len(words) > 18:
                    first_line = " ".join(words[:15]) + "?"
                outputs["question"] = first_line

        return outputs
