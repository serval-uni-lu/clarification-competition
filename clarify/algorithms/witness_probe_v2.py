# SPDX-FileCopyrightText: 2026 zhangchuangnankai <zhangchuang@lenovo.com>
#
# SPDX-License-Identifier: MIT

"""WitnessProbe v2 — anchored solve + more aggressive gating.

Fixes two bugs from WitnessProbe v1:
1. **Less conservative gate**: min_expected_gain 0.04 → 0.02, so it asks
   more often (v1 only asked 40% on tiny5, missing HumanEval/3 /6 /8
   that SGP v1 correctly probed).
2. **Anchored solve**: instead of regenerating from scratch (which caused
   HumanEval/10 hallucinated wrong docstring examples despite a QL=3
   answer), the v2 solve template includes the FIRST-GUESS code as an
   anchor. The LLM edits the draft rather than throwing it away.

Team: zhangchuangnankai
Team Members: Zhang Chuang (Lenovo)
Main Contact: zhangchuang@lenovo.com
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

GATE_TEMPLATE = """You are reviewing a coding task before implementing it.

Task (function to implement: `{entry_point}`):
{prompt}

Is there a SINGLE missing fact such that guessing it wrong would make a
correct-looking implementation fail the user's tests?

Rules:
- Ignore naming, style, and anything you can reasonably infer from the
  description or examples.  Only a genuine behavioural fork counts.
- Do NOT ask about algorithm, test cases, or implementation details.
- You MUST provide a WITNESS: one concrete input ({entry_point}(...))
  whose output differs under the two interpretations of the missing fact.

Reply with exactly one JSON object:
{{"ask": true, "question": "...?", "witness": "{entry_point}(...)", "expected_gain": 0.0}}
or:
{{"ask": false}}
""".strip()

# v2 ANCHORED solve — include first-guess code as draft
FIRST_SOLVE_TEMPLATE = """Implement `{entry_point}` as a self-contained Python module.

{prompt}

Enclose your solution in ```python and ```.
""".strip()

ANCHORED_SOLVE_TEMPLATE = """Here is your FIRST-GUESS implementation of `{entry_point}`:

```python
{draft_code}
```

Clarification received:
Q: {question}
A: {answer}

Revise your first-guess implementation so it follows the answer exactly.
Keep the overall structure — just fix the parts the answer changes.
Enclose your solution in ```python and ```.
""".strip()

REPAIR_TEMPLATE = """Your response did not contain a valid fenced Python
solution for `{entry_point}`.  Return only the complete implementation
enclosed in ```python and ```.
""".strip()


def _extract_json(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for source in re.findall(r"```json\s*\n(.*?)```", text, re.S) + [text]:
        for match in list(re.finditer(r"\{", source))[:32]:
            try:
                value, _ = decoder.raw_decode(source[match.start():])
                if isinstance(value, dict):
                    return value
            except (ValueError, RecursionError):
                continue
    return {}


def _valid_question(decision: dict, min_gain: float, entry_point: str) -> tuple[str, str]:
    if decision.get("ask") is not True:
        return "", ""
    q = (decision.get("question") or "").strip()
    w = (decision.get("witness") or "").strip()
    gain = decision.get("expected_gain", 0)
    if not q.endswith("?") or q.count("?") != 1 or "\n" in q or len(q) > 400:
        return "", ""
    if entry_point not in w or len(w) > 200:
        return "", ""
    if not isinstance(gain, (int, float)) or isinstance(gain, bool):
        return "", ""
    if gain < min_gain:
        return "", ""
    return q, w


def _usable(response: str, entry_point: str) -> bool:
    try:
        code = _validate_and_parse_evalplus_result(response)
    except ValueError:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point
        for n in ast.walk(tree)
    )


def _solve_anchorless(env, prompt, entry_point, attempts):
    """Generate first-guess code (no anchor — this becomes the draft)."""
    messages = [{"role": "user", "content": FIRST_SOLVE_TEMPLATE.format(
        prompt=prompt, entry_point=entry_point)}]
    last = ""
    for i in range(attempts + 1):
        try:
            last = env.llm(messages)
        except Exception:
            break
        if _usable(last, entry_point):
            return last
        if i == attempts:
            break
        messages += [
            {"role": "assistant", "content": last},
            {"role": "user", "content": REPAIR_TEMPLATE.format(entry_point=entry_point)},
        ]
    return last


def _solve_anchored(env, entry_point, draft_code, question, answer, attempts):
    """Revise draft_code using clarification answer — anchored solve."""
    messages = [{"role": "user", "content": ANCHORED_SOLVE_TEMPLATE.format(
        entry_point=entry_point, draft_code=draft_code[:3000],
        question=question, answer=answer)}]
    last = ""
    for i in range(attempts + 1):
        try:
            last = env.llm(messages)
        except Exception:
            break
        if _usable(last, entry_point):
            return last
        if i == attempts:
            break
        messages += [
            {"role": "assistant", "content": last},
            {"role": "user", "content": REPAIR_TEMPLATE.format(entry_point=entry_point)},
        ]
    return last


class WitnessProbe(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "repair_attempts": 2,
        "min_expected_gain": 0.02,  # v2: less conservative than v1's 0.04
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]
        prompt = problem["prompt"]

        # Always generate a first guess — this is our anchor
        draft_code = _solve_anchorless(env, prompt, entry_point, self.config["repair_attempts"])

        question, _ = "", ""
        if env.can_ask():
            try:
                verdict = env.llm(GATE_TEMPLATE.format(
                    prompt=prompt, entry_point=entry_point))
            except Exception:
                verdict = ""
            question, _ = _valid_question(
                _extract_json(verdict), self.config["min_expected_gain"], entry_point)

        if question and env.can_ask():
            try:
                answer = env.ask_human(question)
                # Anchored revise — not from-scratch regeneration!
                code = _solve_anchored(
                    env, entry_point, draft_code, question, answer,
                    self.config["repair_attempts"])
            except TooManyQuestionException:
                code = draft_code
        else:
            code = draft_code

        return code
