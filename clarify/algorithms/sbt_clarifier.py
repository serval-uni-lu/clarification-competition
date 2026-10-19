# SPDX-FileCopyrightText: 2026 Lu Zhang <zhanglu1913@gmail.com>
#
# SPDX-License-Identifier: MIT

"""SBTClarifier.

Placeholder for registration. The final algorithm will replace this file
before the submission deadline.

Team: SBT
Team Members: Lu Zhang
Main Contact: zhanglu1913@gmail.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment


class SBTClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        # Placeholder for registration: returns a stub implementation, no model calls.
        return f"def {problem['entry_point']}(*args, **kwargs):\n    return None\n"
