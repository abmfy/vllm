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

_QWEN_XML_UNRENDERABLE = ("</parameter>", "<parameter=", "</function>")


def _qwen_xml_value(value: Any) -> str:
    rendered = value if isinstance(value, str) else json.dumps(value)
    for marker in _QWEN_XML_UNRENDERABLE:
        if marker in rendered:
            raise ValueError(f"argument value contains unrenderable {marker!r}")
    return rendered


def _render_args(spec: ModelFormatSpec, arguments: dict[str, Any]) -> str:
    assert spec.tool_calls is not None
    if spec.tool_calls.args_encoding == "qwen_xml":
        return "".join(
            f"<parameter={key}>\n{_qwen_xml_value(value)}\n</parameter>"
            for key, value in arguments.items()
        )
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
        assert spec.tool_calls is not None
        rendered = [
            render_tool_call(spec, name, arguments) for name, arguments in tool_calls
        ]
        if content:
            parts.append(spec.tool_calls.separator)
        parts.append(spec.tool_calls.separator.join(rendered))
    return "".join(parts)


def _jinja_str(value: str) -> str:
    return "{{ " + json.dumps(value, ensure_ascii=False) + " }}"


def to_reference_template(spec: ModelFormatSpec) -> str:
    """Jinja fragment rendering an assistant message body from the spec.

    The fragment consumes a ``message`` with optional ``reasoning_content``,
    ``content``, and ``tool_calls`` (OpenAI-shaped: ``function.name`` /
    ``function.arguments`` dict). It renders byte-identically to
    :func:`render_assistant_turn`.
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
        if tool.args_encoding == "qwen_xml":
            args_fragment = (
                "{%- for key, value in tool_call.function.arguments.items() %}"
                "{{ '<parameter=' + key + '>\\n' }}"
                "{{ value if value is string else value | tojson }}"
                "{{ '\\n</parameter>' }}"
                "{%- endfor %}"
            )
        else:
            args_fragment = "{{ tool_call.function.arguments | tojson }}"
        lines.append(
            "{%- for tool_call in message.tool_calls or [] %}"
            "{%- if loop.first and message.content %}"
            + _jinja_str(tool.separator)
            + "{%- elif not loop.first %}"
            + _jinja_str(tool.separator)
            + "{%- endif %}"
            + _jinja_str(tool.call_begin)
            + _jinja_str(tool.name_prefix)
            + "{{ tool_call.function.name }}"
            + _jinja_str(tool.name_suffix)
            + args_fragment
            + _jinja_str(tool.call_end)
            + "{%- endfor %}"
        )
    return "".join(lines)
