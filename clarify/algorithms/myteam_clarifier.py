# SPDX-FileCopyrightText: 2026 Yunhao Zhang <yunhaozhang900@gmail.com>
#
# SPDX-License-Identifier: MIT

"""MyTeamClarifier.

Registration-only placeholder. This stub returns a dummy implementation
without calling a language model or asking clarification questions.
It will be replaced by the final algorithm before the submission deadline.

Team: Hitsz Hago
Team Members: Yunhao Zhang, Jiahao Zhang, Haoman Li, Haojia Xiang, Lianghao Xia, Liqiang Nie
Main Contact: yunhaozhang900@gmail.com
"""

from clarify.baselines.base import ClarificationAlgorithmBase


class MyTeamClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {}

    def run(self, env, problem) -> str:
        entry_point = problem["entry_point"]
        return f"def {entry_point}(*args, **kwargs):\n    return None\n"
