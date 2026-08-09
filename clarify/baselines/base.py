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