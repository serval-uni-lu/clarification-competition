from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from clarify.env import ClarificationEnvironment


class ClarificationAlgorithmBase(ABC):

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config : dict[str, Any] = dict(config or {})

    @abstractmethod
    def run(self, env: ClarificationEnvironment, problem : dict[str, str]) -> str:
        pass


class BatchedClarificationAlgorithmBase(ClarificationAlgorithmBase):
    """
    The class can be used to implement a custom batching logic.
    IMPORTANT: Implementing `batch_run` will cause side-stepping the regular parallization.
    """

    @abstractmethod
    def batch_run(self, envs: list[ClarificationEnvironment], problems: list[dict[str, str]]) -> list[str]:
        pass