"""Common interface for trajectory routers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any


Request = dict[str, Any]


class AbstractRouter(ABC):
    """Interface implemented by model-routing strategies."""

    @abstractmethod
    def route_trajectory(self, calls: Sequence[Request]) -> list[str]:
        """Return one model id for every call in a reconstructed trajectory."""

    @abstractmethod
    def run(self, export: str | Path = "export") -> None:
        """Route an export and write the strategy's evaluation artifacts."""
