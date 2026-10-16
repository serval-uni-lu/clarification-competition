# SPDX-FileCopyrightText: 2026 kelvin715 <yzhsk99@gmail.com>
#
# SPDX-License-Identifier: MIT

"""SkyWalker Clarification.

Registration placeholder for team SkyWalker. This version generates the implementation
directly with `env.llm` and does not ask a clarifying question. The final algorithm will
replace this file before the submission deadline.

Team: SkyWalker
Team Members: kelvin715
Main Contact: yzhsk99@gmail.com
"""

from __future__ import annotations

import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment

SOLVE_TEMPLATE = """Implement the Python function `{entry_point}` for the task below.

{prompt}

Reply with one ```python code block containing the complete implementation and its imports."""

CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class SkyWalkerClarification(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem.get("entry_point") or "solution"
        prompt = (problem.get("prompt") or "").strip()
        content = SOLVE_TEMPLATE.format(entry_point=entry_point, prompt=prompt)
        try:
            response = env.llm([{"role": "user", "content": content}]) or ""
        except Exception:
            response = ""
        blocks = CODE_BLOCK.findall(response)
        code = blocks[-1] if blocks else response
        return f"```python\n{code.strip()}\n```"
