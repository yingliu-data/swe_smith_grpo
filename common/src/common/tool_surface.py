from __future__ import annotations

import json
import re
from typing import Any

from agent import ToolCall, ToolResult
from agent.docker_env import DockerEnvironment

FIXED_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repo. Path is relative to repo root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply a unified diff. May not target test files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "string", "description": "unified diff"},
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file. May not target test files.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the task's test_command; returns stdout/stderr and exit code.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

VALID_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "edit_file", "delete_file", "run_tests"})


class ToolSurfaceError(ValueError):
    pass


_SWE_AGENT_RE = re.compile(r"^TOOL\|([a-z_]+)\|(.*)$", re.DOTALL)


def parse_tool_call(model_output: str) -> ToolCall:
    """Parse model output to a ToolCall.

    Accepts two formats:
      * OpenAI tool-call JSON: ``{"name": "...", "arguments": {...}}``
      * SWE-agent pipe-form: ``TOOL|<name>|<json-args>``
    Anything else raises ToolSurfaceError.
    """
    text = model_output.strip()
    if not text:
        raise ToolSurfaceError("empty model output")
    if text.startswith("{"):
        return _parse_json(text)
    m = _SWE_AGENT_RE.match(text)
    if not m:
        raise ToolSurfaceError("unrecognised tool-call format")
    name = m.group(1)
    if name not in VALID_TOOL_NAMES:
        raise ToolSurfaceError(f"unknown tool: {name!r}")
    raw = m.group(2).strip()
    args: dict[str, Any] = {}
    if raw:
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolSurfaceError(f"invalid json args: {exc}") from exc
    return ToolCall(name=name, arguments=args)


def _parse_json(text: str) -> ToolCall:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolSurfaceError(f"invalid json: {exc}") from exc
    name = obj.get("name") or obj.get("tool") or obj.get("function", {}).get("name")
    args: Any = obj.get("arguments") or obj.get("args") or obj.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if name not in VALID_TOOL_NAMES:
        raise ToolSurfaceError(f"unknown tool: {name!r}")
    return ToolCall(name=name, arguments=args)


async def dispatch(tool_call: ToolCall, env: DockerEnvironment) -> ToolResult:
    if tool_call.name not in VALID_TOOL_NAMES:
        return ToolResult(name=tool_call.name, ok=False, error="tool not in fixed surface")
    return await env.step(tool_call)
