# SPDX-FileCopyrightText: 2026 杨嵩
#
# SPDX-License-Identifier: MIT

"""EDYClarifier.

Initial registration draft for the Clarification Challenge.
The final clarification strategy is still under development.

Team: EDY
Team Members: 曹向前 (Cao Xiangqian, Inner Mongolia University), 杨嵩 (Yang Song, Suzhou Heshuju Information Technology Co., Ltd.), 李凯 (Li Kai, Suzhou Heshuju Information Technology Co., Ltd.)
Main Contact: 杨嵩(Yang Song) <song.yang@core-dt.com>
"""

from clarify.baselines import ClarificationAlgorithmBase


class EDYClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {}

    def run(self, env, problem) -> str:
        raise NotImplementedError("Registration draft; final implementation pending.")
