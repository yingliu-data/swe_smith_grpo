from abc import ABC, abstractmethod
from pathlib import Path

from .models import EvaluationResult, TaskSpec, ToolCall, ToolResult


class Environment(ABC):

    def __init__(self, workspace_root: str | Path, command_timeout_seconds: int = 120):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.command_timeout_seconds = command_timeout_seconds
        self.task: TaskSpec | None = None

    @abstractmethod
    def step(self, tool_call: ToolCall) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def read_file(self, path: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def edit_file(self, path: str, patch: str) -> ToolResult:
        raise NotImplementedError
    
    @abstractmethod
    def delete_file(self, path: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError
