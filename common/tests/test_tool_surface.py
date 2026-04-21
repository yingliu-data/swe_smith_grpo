from __future__ import annotations

import pytest

from common.tool_surface import FIXED_TOOL_DEFS, ToolSurfaceError, parse_tool_call


def test_fixed_surface_has_exactly_four_tools():
    names = {t["function"]["name"] for t in FIXED_TOOL_DEFS}
    assert names == {"read_file", "edit_file", "delete_file", "evaluate"}


def test_parse_openai_json():
    call = parse_tool_call('{"name":"read_file","arguments":{"path":"mod.py"}}')
    assert call.name == "read_file"
    assert call.arguments == {"path": "mod.py"}


def test_parse_openai_function_shape():
    call = parse_tool_call(
        '{"function":{"name":"edit_file","arguments":"{\\"path\\":\\"a.py\\",\\"patch\\":\\"diff\\"}"}}'
    )
    assert call.name == "edit_file"
    assert call.arguments == {"path": "a.py", "patch": "diff"}


def test_parse_swe_agent_pipe_form():
    call = parse_tool_call('TOOL|evaluate|{}')
    assert call.name == "evaluate"
    assert call.arguments == {}


def test_parse_rejects_unknown_tool():
    with pytest.raises(ToolSurfaceError):
        parse_tool_call('{"name":"curl","arguments":{"url":"http://evil"}}')


def test_parse_rejects_empty():
    with pytest.raises(ToolSurfaceError):
        parse_tool_call("   ")
