"""Mitigation for a Gemma tool-call block that does not parse — unterminated because the reply hit the token limit
mid-arguments, or with corrupted arguments. Instead of leaking raw ``<|tool_call>`` markup into the reply text, the
parser repairs an offered call into a structured call (salvaging a truncated final string) or hides the markup."""

from __future__ import annotations

import json

from tensorfold.server.tools import parse_tool_calls_from_content

WRITE = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "limit": {"type": "integer"},
    "filePath": {"type": "string"}}}}}]
GLOB = [{"type": "function", "function": {"name": "glob", "parameters": {"type": "object", "properties": {
    "pattern": {"type": "string"}}}}}]


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


def test_terminated_block_with_unbalanced_string_is_repaired():
    # the observed glob leak: a terminated block whose pattern string opened with <|"|> but never closed (odd count)
    reply = '<|tool_call>call:glob{pattern:<|"|>*/}<tool_call|>'
    content, calls = parse_tool_calls_from_content(reply, GLOB)
    assert "<|tool_call>" not in content and content == ""
    assert calls and calls[0]["function"]["name"] == "glob" and _args(calls[0])["pattern"] == "*/"


def test_text_after_a_repaired_terminated_block_is_kept():
    reply = 'Found it.\n<|tool_call>call:glob{pattern:<|"|>*/}<tool_call|>\nok'
    content, calls = parse_tool_calls_from_content(reply, GLOB)
    assert content == "Found it.\n\nok" and calls[0]["function"]["name"] == "glob"


def test_terminated_block_with_a_bare_keyword_stays_as_text():
    # a bare key, no value, no salvageable args -> left as text, matching upstream's deliberate behaviour
    reply = "<|tool_call>call:write{path}<tool_call|>"
    assert parse_tool_calls_from_content(reply, WRITE) == (reply, None)


def test_terminated_balanced_but_degenerate_value_is_repaired():
    # balanced <|"|> (even), but the model emitted a value as a bare degenerate token instead of a <|"|> string
    reply = '<|tool_call>call:write{content:<|"|>const x = 1;<|"|>,filePath:mapsto_path_now}<tool_call|>'
    content, calls = parse_tool_calls_from_content(reply, WRITE)
    assert "<|tool_call>" not in content and content == ""
    args = _args(calls[0])
    assert calls[0]["function"]["name"] == "write"
    assert args["content"] == "const x = 1;" and args["filePath"] == "mapsto_path_now"


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
