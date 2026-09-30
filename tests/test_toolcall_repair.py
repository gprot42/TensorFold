"""Mitigation for a Gemma tool-call block that does not parse — unterminated because the reply hit the token limit
mid-arguments, or with corrupted arguments. Instead of leaking raw ``<|tool_call>`` markup into the reply text, the
parser repairs an offered call into a structured call (salvaging a truncated final string) or hides the markup."""

from __future__ import annotations

import json

from tensorfold.server.tools import parse_tool_calls_from_content

WRITE = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "limit": {"type": "integer"}}}}}]


def _args(call):
    return json.loads(call["function"]["arguments"])


def test_unterminated_call_is_repaired_into_a_structured_call():
    # the reply ran out of tokens after opening the content string and never closed it or the block
    reply = '<|tool_call>call:write{path:<|"|>game.html<|"|>,content:<|"|><!DOCTYPE html>\n<html><body>'
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert "<|tool_call>" not in content and content == ""
    assert calls and calls[0]["function"]["name"] == "write"
    args = _args(calls[0])
    assert args["path"] == "game.html" and args["content"].startswith("<!DOCTYPE html>")


def test_a_terminated_block_is_left_to_normal_parsing():
    # scope boundary: repair only touches UNTERMINATED blocks. A block that closed with <tool_call|> but has
    # malformed args stays as text, matching the deliberate existing behaviour (a small malformed call is content).
    reply = '<|tool_call>call:write{path:<|"|>a.html<|"|>,content:<|"|>x}|}{|}:{}<tool_call|>'
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert calls is None and content == reply       # untouched: terminated, so out of repair scope


def test_text_before_the_broken_call_is_kept():
    reply = 'Let me write it.\n<|tool_call>call:write{path:<|"|>g.html<|"|>,content:<|"|><html>'
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert content == "Let me write it." and calls[0]["function"]["name"] == "write"


def test_a_broken_call_to_an_unoffered_tool_is_hidden_not_leaked():
    reply = '<|tool_call>call:danger{cmd:<|"|>rm -rf /<|"|>'      # 'danger' is not an offered tool
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert content == "" and calls is None                       # markup hidden, no call invented


def test_plain_text_that_merely_mentions_the_marker_is_untouched():
    reply = "The <|tool_call> marker begins a Gemma call."        # not call-shaped after the marker
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert content == reply and calls is None


def test_bare_and_numeric_values_survive_a_repair():
    reply = '<|tool_call>call:write{limit:20,path:<|"|>a.py<|"|>,content:<|"|>partial'
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    args = _args(calls[0])
    assert args["limit"] == 20 and args["path"] == "a.py" and args["content"] == "partial"
    assert content == ""


def test_a_complete_call_still_parses_normally_and_is_untouched():
    call = '<|tool_call>call:write{path:<|"|>a.py<|"|>,content:<|"|>hi<|"|>}<tool_call|>'
    content, calls = parse_tool_calls_from_content("Writing." + call, WRITE)
    assert content == "Writing." and _args(calls[0]) == {"path": "a.py", "content": "hi"}


def test_no_tools_offered_leaves_text_alone():
    reply = '<|tool_call>call:write{path:<|"|>a<|"|>'
    assert parse_tool_calls_from_content(reply, []) == (reply, None)
