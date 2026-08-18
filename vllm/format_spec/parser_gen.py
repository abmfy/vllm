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
        # A start marker mid-content re-enters reasoning (GLM/DSML style).
        transitions[(ParserState.CONTENT, "THINK_START")] = Transition(
            ParserState.REASONING,
            (EventType.REASONING_START,),
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


def _arg_key_value_xml_tool_config(spec: ModelFormatSpec) -> tuple[dict, dict, dict]:
    """FSM for GLM-style calls: ``<tool_call>NAME<arg_key>K</arg_key>...``.

    The name has no suffix terminal — it ends at the first ``<arg_key>``
    or at ``call_end`` (zero-argument calls). Mirrors the hand-written
    ``glm47_moe_config`` topology.
    """
    from vllm.parser.glm47_moe import _glm47_arg_converter

    tool = spec.tool_calls
    assert tool is not None
    terminals = {
        "TOOL_START": tool.trigger,
        "TOOL_END": _CLOSING_TAG_RE.findall(tool.call_end)[-1],
        "ARG_KEY_START": "<arg_key>",
        "ARG_KEY_END": "</arg_key>",
        "ARG_VALUE_START": "<arg_value>",
        "ARG_VALUE_END": "</arg_value>",
    }
    transitions = {
        (ParserState.REASONING, "TOOL_START"): Transition(
            ParserState.TOOL_NAME,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        ),
        (ParserState.CONTENT, "TOOL_START"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_NAME, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        ),
    }
    transitions.update(
        {
            (ParserState.TOOL_ARGS, terminal): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            )
            for terminal in (
                "ARG_KEY_START",
                "ARG_KEY_END",
                "ARG_VALUE_START",
                "ARG_VALUE_END",
            )
        }
    )
    options = {
        "arg_converter": _glm47_arg_converter,
        "tool_args_json": False,
        "validate_tool_names": True,
    }
    return terminals, transitions, options


_MINIMAX_NS = "]<]minimax[>["


def _minimax_ns_args_converter(raw_args: str, partial: bool) -> str:
    """Parse ``{NS}<K>V{NS}</K>`` element blocks (nested) into JSON.

    Values stay raw strings at the leaves; sibling ``<item>`` elements
    collapse into a list. Unterminated elements contribute their partial
    text when ``partial`` is set.
    """
    root: dict = {}
    stack: list[tuple[str, dict, list[str]]] = [("", root, [])]
    for chunk in raw_args.split(_MINIMAX_NS)[1:]:
        closing = chunk.startswith("</")
        gt = chunk.find(">")
        if gt < 0:
            continue
        tag = chunk[2:gt] if closing else chunk[1:gt]
        text = chunk[gt + 1 :]
        if closing:
            if len(stack) > 1 and stack[-1][0] == tag:
                name, children, texts = stack.pop()
                value = children if children else "".join(texts)
                _ns_insert(stack[-1][1], name, value)
        else:
            stack.append((tag, {}, [text]))
    if partial:
        while len(stack) > 1:
            name, children, texts = stack.pop()
            value = children if children else "".join(texts)
            _ns_insert(stack[-1][1], name, value)
    return json.dumps(_ns_finalize(root), ensure_ascii=False)


def _ns_insert(container: dict, name: str, value) -> None:
    if name in container:
        existing = container[name]
        if isinstance(existing, list):
            existing.append(value)
        else:
            container[name] = [existing, value]
    else:
        container[name] = value


def _ns_finalize(node):
    if isinstance(node, dict):
        if set(node) == {"item"}:
            items = node["item"] if isinstance(node["item"], list) else [node["item"]]
            return [_ns_finalize(item) for item in items]
        return {key: _ns_finalize(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_ns_finalize(item) for item in node]
    return node


def _sectioned_tool_config(spec: ModelFormatSpec) -> tuple[dict, dict, dict]:
    """FSM for section-wrapped calls (DSML, MiniMax-M3 namespace XML).

    ``{section_begin} {name_prefix}NAME{name_suffix}ARGS{call_end} ...
    {section_end}`` — mirrors the hand-written ``deepseek_v4_config``
    topology.
    """
    tool = spec.tool_calls
    assert tool is not None
    if not tool.section_begin:
        raise ValueError(f"{spec.name}: sectioned encoding requires section markers")
    terminals = {
        "TOOL_START": tool.trigger,
        "TOOL_END": tool.section_end.strip(),
        "INVOKE_PREFIX": tool.name_prefix,
        "INVOKE_NAME_END": tool.name_suffix.rstrip("\n"),
        "INVOKE_END": tool.call_end.strip(),
    }
    transitions = {
        (ParserState.REASONING, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (EventType.REASONING_END,),
        ),
        (ParserState.CONTENT, "TOOL_START"): Transition(ParserState.TOOL_PREAMBLE, ()),
        (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(ParserState.CONTENT, ()),
        (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
            ParserState.TOOL_ARGS, ()
        ),
        (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(ParserState.CONTENT, ()),
    }
    if tool.args_encoding == "dsml":
        terminals["PARAM_CLOSE"] = "</｜DSML｜parameter>"
        from vllm.parser.deepseek_v4 import _dsml_arg_converter

        options = {
            "arg_converter": _dsml_arg_converter,
            "arg_structural_chars": frozenset(">"),
            "strip_content_whitespace_with_tools": False,
            "tool_args_json": False,
        }
    else:
        options = {
            "arg_converter": _minimax_ns_args_converter,
            "tool_args_json": False,
        }
    return terminals, transitions, options


_TOOL_CONFIG_BUILDERS = {
    "json": _json_tool_config,
    "qwen_xml": _qwen_xml_tool_config,
    "arg_key_value_xml": _arg_key_value_xml_tool_config,
    "dsml": _sectioned_tool_config,
    "minimax_ns_xml": _sectioned_tool_config,
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

    if spec.reasoning is not None and spec.reasoning.forced:
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
