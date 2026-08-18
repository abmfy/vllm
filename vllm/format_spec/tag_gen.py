# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate xgrammar structural tags from a :class:`ModelFormatSpec`.

The tag is built from the same literals that drive the parser FSM and the
wire renderer; the round-trip tests in ``tests/format_spec`` keep the
three artifacts in byte-level agreement. Intended to replace the
per-model builders in ``vllm/tool_parsers/structural_tag_registry.py``
for spec'd models once wired into the registry.
"""

from __future__ import annotations

from typing import Any

from xgrammar.structural_tag import (
    AnyTextFormat,
    Format,
    JSONSchemaFormat,
    QwenXMLParameterFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.format_spec.spec import ModelFormatSpec


def _args_format(spec: ModelFormatSpec, parameters: dict[str, Any] | bool) -> Format:
    assert spec.tool_calls is not None
    encoding = spec.tool_calls.args_encoding
    if encoding == "qwen_xml":
        # A bare object schema without properties would reject every value.
        if parameters is True:
            parameters = {"type": "object", "additionalProperties": True}
        elif isinstance(parameters, dict) and "properties" not in parameters:
            parameters = {**parameters, "additionalProperties": True}
        return QwenXMLParameterFormat(json_schema=parameters)
    if encoding == "json":
        return JSONSchemaFormat(json_schema=parameters)
    raise ValueError(f"unsupported args_encoding: {encoding!r}")


def tool_tag_formats(
    spec: ModelFormatSpec,
    tools: list[tuple[str, dict[str, Any] | bool]],
) -> list[TagFormat]:
    """One :class:`TagFormat` per (name, parameter-schema) tool."""
    tool_shape = spec.tool_calls
    assert tool_shape is not None
    return [
        TagFormat(
            begin=(
                tool_shape.call_begin
                + tool_shape.name_prefix
                + name
                + tool_shape.name_suffix
            ),
            content=_args_format(spec, parameters),
            end=tool_shape.call_end,
        )
        for name, parameters in tools
    ]


def tool_structural_tag(
    spec: ModelFormatSpec,
    tools: list[tuple[str, dict[str, Any] | bool]],
    tool_choice: str = "auto",
) -> StructuralTag:
    """Constrained tool calling in the model's native wire format.

    Args:
        spec: The model format description.
        tools: (function name, parameter JSON schema) pairs; a schema of
            True leaves the arguments unconstrained.
        tool_choice: "auto" (tool calls optional, triggered), "forced"
            (exactly one call), or "required" (at least one call).
    """
    if spec.tool_calls is None:
        raise ValueError(f"spec {spec.name!r} declares no tool-call format")
    tags = tool_tag_formats(spec, tools)

    suffix: Format
    if tool_choice == "auto":
        suffix = (
            TriggeredTagsFormat(triggers=[spec.tool_calls.trigger], tags=tags)
            if tags
            else AnyTextFormat()
        )
    elif tool_choice == "forced":
        suffix = TagsWithSeparatorFormat(
            tags=tags,
            separator=spec.tool_calls.separator,
            at_least_one=True,
            stop_after_first=True,
        )
    elif tool_choice == "required":
        suffix = TagsWithSeparatorFormat(
            tags=tags,
            separator=spec.tool_calls.separator,
            at_least_one=True,
        )
    else:
        raise ValueError(f"unsupported tool_choice: {tool_choice!r}")

    return StructuralTag(format=suffix)
