"""
Dummy implementation to showcase custom batching
"""

from clarify.baselines.base import BatchedClarificationAlgorithmBase
from clarify.baselines.direct import LLMClarification
from clarify.env import ClarificationEnvironment


class BatchedLLMClarification(LLMClarification, BatchedClarificationAlgorithmBase):

    def batch_run(self, envs: list[ClarificationEnvironment], problems: list[dict[str, str]]) -> list[str]:
        return [
            self.run(env, problem) for env, problem in zip(envs, problems, strict = True)
        ]