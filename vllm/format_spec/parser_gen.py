# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a :class:`ParserEngineConfig` from a :class:`ModelFormatSpec`.

The per-family FSM topology lives here once, keyed by ``args_encoding``;
the spec contributes only the model-specific literals. The generated
config flows through the existing ``ParserEngine`` and ``make_adapters``
machinery, so one spec yields both the reasoning parser and the tool
parser with no per-model parser code.
"""

from __future__ import annotations

import json

import regex as re

from vllm.format_spec.spec import ModelFormatSpec
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.qwen3 import _qwen3_arg_converter

_CLOSING_TAG_RE = re.compile(r"\S+")


def _reasoning_terminals(spec: ModelFormatSpec) -> dict[str, str]:
    if spec.reasoning is None:
        return {}
    terminals = {"THINK_END": spec.reasoning.end}
    if spec.reasoning.start is not None:
        terminals["THINK_START"] = spec.reasoning.start
    return terminals


def _reasoning_transitions(
    spec: ModelFormatSpec,
) -> dict[tuple[ParserState, str], Transition]:
    if spec.reasoning is None:
        return {}
    transitions = {
        (ParserState.REASONING, "THINK_END"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        # Absorb a duplicate end marker emitted after the transition.
        (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT, ()),
    }
    if spec.reasoning.start is not None:
        transitions[(ParserState.REASONING, "THINK_START")] = Transition(
            ParserState.REASONING, ()
        )
    return transitions


def _json_args_converter(raw_args: str, partial: bool) -> str:
    """Extract the arguments value from an accumulated JSON call body.

    The body is the arguments value followed by the wrapper object's
    closing brace and separator whitespace. Outputs are byte-exact
    prefixes of the argument text (prefix-stable, as streaming deltas
    require): incomplete JSON is passed through as-is, and once
    ``raw_decode`` finds the complete first value the body is trimmed to
    exactly that value, dropping the wrapper remainder.
    """
    raw = raw_args.lstrip()
    if not raw:
        return ""
    try:
        _, end = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return raw if partial else raw_args
    return raw[:end]


def _json_tool_config(spec: ModelFormatSpec) -> tuple[dict, dict, dict]:
    """FSM for tool calls whose arguments are a plain JSON object.

    Only the wrapper markers are lexed as terminals (so they keep working
    when emitted as special tokens); the argument body is recovered by a
    JSON-aware converter instead of byte-exact literals.
    """
    tool = spec.tool_calls
    assert tool is not None
    closers = _CLOSING_TAG_RE.findall(tool.call_end)
    if not closers:
        raise ValueError(f"call_end {tool.call_end!r} has no closing marker")
    terminals = {
        "TOOL_START": tool.trigger,
        "TOOL_END": closers[-1],
        "NAME_PREFIX": tool.name_prefix,
        "NAME_SUFFIX": tool.name_suffix,
    }
    tool_start = Transition(
        ParserState.TOOL_PREAMBLE,
        (EventType.REASONING_END, EventType.TOOL_CALL_START),
    )
    transitions = {
        (ParserState.CONTENT, "TOOL_START"): tool_start,
        (ParserState.REASONING, "TOOL_START"): tool_start,
        (ParserState.TOOL_PREAMBLE, "NAME_PREFIX"): Transition(
            ParserState.TOOL_NAME, ()
        ),
        (ParserState.TOOL_NAME, "NAME_SUFFIX"): Transition(ParserState.TOOL_ARGS, ()),
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (EventType.TOOL_CALL_START,),
        ),
    }
    options = {"arg_converter": _json_args_converter, "tool_args_json": False}
    return terminals, transitions, options


def _qwen_xml_tool_config(spec: ModelFormatSpec) -> tuple[dict, dict, dict]:
    """FSM for Qwen3-style XML tool calls (``<parameter=K>V</parameter>``).

    ``call_end`` is split into its closing tags (e.g. ``</function>`` and
    ``</tool_call>``) so that consecutive calls omitting the outer closer
    are still parsed, mirroring the hand-written Qwen3 topology.
    """
    tool = spec.tool_calls
    assert tool is not None
    closers = _CLOSING_TAG_RE.findall(tool.call_end)
    if len(closers) != 2:
        raise ValueError(
            f"qwen_xml call_end must contain two closing tags, got {closers}"
        )
    func_end, tool_end = closers
    terminals = {
        "TOOL_START": tool.trigger,
        "TOOL_END": tool_end,
        "FUNC_PREFIX": tool.name_prefix,
        "FUNC_END": func_end,
        "NAME_SUFFIX": tool.name_suffix.rstrip("\n"),
        "PARAM_START": "<parameter=",
        "PARAM_END": "</parameter>",
    }
    tool_start = Transition(
        ParserState.TOOL_PREAMBLE,
        (EventType.REASONING_END, EventType.TOOL_CALL_START),
    )
    transitions = {
        (ParserState.CONTENT, "TOOL_START"): tool_start,
        (ParserState.REASONING, "TOOL_START"): tool_start,
        (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME, ()
        ),
        (ParserState.TOOL_NAME, "NAME_SUFFIX"): Transition(ParserState.TOOL_ARGS, ()),
        (ParserState.TOOL_NAME, "FUNC_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(ParserState.CONTENT, ()),
        (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
    }
    options = {
        "arg_converter": _qwen3_arg_converter,
        "tool_args_json": False,
        "strip_trailing_reasoning_whitespace": False,
    }
    return terminals, transitions, options


_TOOL_CONFIG_BUILDERS = {
    "json": _json_tool_config,
    "qwen_xml": _qwen_xml_tool_config,
}


def to_parser_engine_config(
    spec: ModelFormatSpec,
    thinking: bool = True,
) -> ParserEngineConfig:
    """Build the streaming parser FSM for ``spec``.

    Args:
        spec: The model format description.
        thinking: Whether this request starts in the reasoning state.
    """
    terminals = _reasoning_terminals(spec)
    transitions = _reasoning_transitions(spec)
    options: dict = {}

    if spec.tool_calls is not None:
        tool_terminals, tool_transitions, options = _TOOL_CONFIG_BUILDERS[
            spec.tool_calls.args_encoding
        ](spec)
        terminals.update(tool_terminals)
        transitions.update(tool_transitions)

    if spec.reasoning is not None and (
        spec.reasoning.forced or spec.reasoning.start_in_prompt
    ):
        thinking = True
    start_in_reasoning = thinking and spec.reasoning is not None

    token_id_terminal_names = ("THINK_START", "THINK_END", "TOOL_START", "TOOL_END")
    token_id_terminals = {
        name: terminals[name] for name in token_id_terminal_names if name in terminals
    }

    return ParserEngineConfig(
        name=spec.name,
        initial_state=(
            ParserState.REASONING if start_in_reasoning else ParserState.CONTENT
        ),
        terminals=terminals,
        token_id_terminals=token_id_terminals,
        transitions=transitions,
        **options,
    )
