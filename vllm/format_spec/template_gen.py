# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire rendering and reference chat-template generation from a spec.

``render_assistant_turn`` is the authoritative renderer of an assistant
turn body: the exact byte sequence the model is expected to generate.
``to_reference_template`` emits the equivalent Jinja fragment, proving the
template's assistant-turn body is mechanically derivable from the spec.
Round-trip tests assert renderer == template == parser == grammar.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vllm.format_spec.spec import ModelFormatSpec

_MINIMAX_NS = "]<]minimax[>["

_UNRENDERABLE = {
    "qwen_xml": ("</parameter>", "<parameter=", "</function>"),
    "arg_key_value_xml": ("</arg_value>", "<arg_key>"),
    "dsml": ("</｜DSML｜parameter>",),
    "minimax_ns_xml": (_MINIMAX_NS,),
}


def _scalar(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _guarded_scalar(encoding: str, value: Any) -> str:
    rendered = _scalar(value)
    for marker in _UNRENDERABLE.get(encoding, ()):
        if marker in rendered:
            raise ValueError(f"argument value contains unrenderable {marker!r}")
    return rendered


def _render_ns_value(value: Any) -> str:
    if isinstance(value, dict):
        return "".join(
            f"{_MINIMAX_NS}<{key}>{_render_ns_value(item)}{_MINIMAX_NS}</{key}>"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return "".join(
            f"{_MINIMAX_NS}<item>{_render_ns_value(item)}{_MINIMAX_NS}</item>"
            for item in value
        )
    return _guarded_scalar("minimax_ns_xml", value)


def _render_args(spec: ModelFormatSpec, arguments: dict[str, Any]) -> str:
    assert spec.tool_calls is not None
    encoding = spec.tool_calls.args_encoding
    if encoding == "qwen_xml":
        return "".join(
            f"<parameter={key}>\n{_guarded_scalar(encoding, value)}\n</parameter>"
            for key, value in arguments.items()
        )
    if encoding == "arg_key_value_xml":
        return "".join(
            f"<arg_key>{key}</arg_key>\n"
            f"<arg_value>{_guarded_scalar(encoding, value)}</arg_value>\n"
            for key, value in arguments.items()
        )
    if encoding == "dsml":
        return "\n".join(
            f'<｜DSML｜parameter name="{key}" '
            f'string="{"true" if isinstance(value, str) else "false"}">'
            f"{_guarded_scalar(encoding, value)}</｜DSML｜parameter>"
            for key, value in arguments.items()
        )
    if encoding == "minimax_ns_xml":
        return _render_ns_value(arguments)
    return json.dumps(arguments, ensure_ascii=False)


def render_tool_call(spec: ModelFormatSpec, name: str, arguments: dict) -> str:
    tool = spec.tool_calls
    assert tool is not None
    return (
        tool.call_begin
        + tool.name_prefix
        + name
        + tool.name_suffix
        + _render_args(spec, arguments)
        + tool.call_end
    )


def render_assistant_turn(
    spec: ModelFormatSpec,
    *,
    reasoning: str | None = None,
    content: str | None = "",
    tool_calls: Sequence[tuple[str, dict[str, Any]]] = (),
) -> str:
    """Render the assistant-turn body exactly as the model generates it."""
    content = content or ""
    parts: list[str] = []
    if reasoning is not None and spec.reasoning is not None:
        if spec.reasoning.emits_start:
            parts.append(spec.reasoning.start or "")
        parts.append(reasoning)
        parts.append(spec.reasoning.end)
    parts.append(content)
    if tool_calls:
        tool = spec.tool_calls
        assert tool is not None
        rendered = tool.separator.join(
            render_tool_call(spec, name, arguments) for name, arguments in tool_calls
        )
        if content:
            parts.append(tool.separator)
        parts.append(tool.section_begin + rendered + tool.section_end)
    return "".join(parts)


def _jinja_str(value: str) -> str:
    return "{{ " + json.dumps(value, ensure_ascii=False) + " }}"


_JINJA_SCALAR = "{{ value if value is string else value | tojson }}"


def _jinja_args_fragment(spec: ModelFormatSpec) -> str:
    """Per-call arguments loop; flat (scalar-valued) arguments only."""
    tool = spec.tool_calls
    assert tool is not None
    loop = "{%- for key, value in tool_call.function.arguments.items() %}"
    end = "{%- endfor %}"
    if tool.args_encoding == "qwen_xml":
        return (
            loop
            + "{{ '<parameter=' + key + '>\\n' }}"
            + _JINJA_SCALAR
            + "{{ '\\n</parameter>' }}"
            + end
        )
    if tool.args_encoding == "arg_key_value_xml":
        return (
            loop
            + "{{ '<arg_key>' + key + '</arg_key>\\n<arg_value>' }}"
            + _JINJA_SCALAR
            + "{{ '</arg_value>\\n' }}"
            + end
        )
    if tool.args_encoding == "dsml":
        return (
            loop
            + "{%- if not loop.first %}{{ '\\n' }}{%- endif %}"
            + "{{ '<｜DSML｜parameter name=\"' + key + '\" string=\"' }}"
            + "{{ 'true' if value is string else 'false' }}{{ '\">' }}"
            + _JINJA_SCALAR
            + "{{ '</｜DSML｜parameter>' }}"
            + end
        )
    if tool.args_encoding == "minimax_ns_xml":
        ns = json.dumps(_MINIMAX_NS)
        return (
            loop
            + "{{ "
            + ns
            + " + '<' + key + '>' }}"
            + _JINJA_SCALAR
            + "{{ "
            + ns
            + " + '</' + key + '>' }}"
            + end
        )
    return "{{ tool_call.function.arguments | tojson }}"


def to_reference_template(spec: ModelFormatSpec) -> str:
    """Jinja fragment rendering an assistant message body from the spec.

    The fragment consumes a ``message`` with optional ``reasoning_content``,
    ``content``, and ``tool_calls`` (OpenAI-shaped: ``function.name`` /
    ``function.arguments`` dict). It renders byte-identically to
    :func:`render_assistant_turn` for flat argument values.
    """
    lines: list[str] = []
    if spec.reasoning is not None:
        start = (
            _jinja_str(spec.reasoning.start)
            if spec.reasoning.emits_start and spec.reasoning.start
            else ""
        )
        lines.append(
            "{%- if message.reasoning_content is defined"
            " and message.reasoning_content is not none %}"
            + start
            + "{{ message.reasoning_content }}"
            + _jinja_str(spec.reasoning.end)
            + "{%- endif %}"
        )
    lines.append('{{ message.content or "" }}')
    if spec.tool_calls is not None:
        tool = spec.tool_calls
        call_fragment = (
            _jinja_str(tool.call_begin)
            + _jinja_str(tool.name_prefix)
            + "{{ tool_call.function.name }}"
            + _jinja_str(tool.name_suffix)
            + _jinja_args_fragment(spec)
            + _jinja_str(tool.call_end)
        )
        lines.append(
            "{%- if message.tool_calls %}"
            "{%- if message.content %}"
            + _jinja_str(tool.separator)
            + "{%- endif %}"
            + (_jinja_str(tool.section_begin) if tool.section_begin else "")
            + "{%- for tool_call in message.tool_calls %}"
            + "{%- if not loop.first %}"
            + _jinja_str(tool.separator)
            + "{%- endif %}"
            + call_fragment
            + "{%- endfor %}"
            + (_jinja_str(tool.section_end) if tool.section_end else "")
            + "{%- endif %}"
        )
    return "".join(lines)
