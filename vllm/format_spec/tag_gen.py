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
    ConstStringFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    PlusFormat,
    QwenXMLParameterFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.format_spec.spec import ModelFormatSpec

_MINIMAX_NS = "]<]minimax[>["


def _normalize_object_schema(parameters: dict[str, Any] | bool) -> dict[str, Any]:
    """A bare object schema without properties would reject every value."""
    if parameters is True:
        return {"type": "object", "additionalProperties": True}
    assert isinstance(parameters, dict)
    if "properties" not in parameters:
        return {**parameters, "additionalProperties": True}
    return parameters


def _minimax_ns_value_format(schema: dict[str, Any]) -> Format:
    kind = schema.get("type")
    if kind == "string":
        return AnyTextFormat(excludes=[_MINIMAX_NS])
    if kind == "object":
        return _minimax_ns_args_format(schema)
    if kind == "array":
        # PlusFormat (not Star): the wire cannot represent an empty array,
        # so the grammar must not admit one the parser misreads as "".
        return PlusFormat(
            content=TagFormat(
                begin=f"{_MINIMAX_NS}<item>",
                content=_minimax_ns_value_format(schema.get("items", {})),
                end=f"{_MINIMAX_NS}</item>",
            )
        )
    if kind in ("integer", "number", "boolean", "null"):
        return JSONSchemaFormat(json_schema=schema)
    raise ValueError(
        f"minimax_ns_xml cannot express schema without a recognized type: {schema!r}"
    )


def _minimax_ns_args_format(parameters: dict[str, Any] | bool) -> Format:
    """Recursive ``{NS}<K>V{NS}</K>`` element grammar from a JSON schema.

    Properties render in schema declaration order (the renderer's order);
    optional properties may be omitted.
    """
    if parameters is True or not isinstance(parameters, dict):
        return AnyTextFormat(excludes=[_MINIMAX_NS])
    properties = parameters.get("properties")
    if not properties:
        return AnyTextFormat(excludes=[_MINIMAX_NS])
    required = set(parameters.get("required", ()))
    elements: list[Format] = []
    for key, subschema in properties.items():
        if key == "item":
            # "item" is the reserved array-element tag; a property with
            # that name would be indistinguishable from a list on parse.
            raise ValueError("minimax_ns_xml cannot express a property named 'item'")
        element: Format = TagFormat(
            begin=f"{_MINIMAX_NS}<{key}>",
            content=_minimax_ns_value_format(subschema),
            end=f"{_MINIMAX_NS}</{key}>",
        )
        if key not in required:
            element = OptionalFormat(content=element)
        elements.append(element)
    return SequenceFormat(elements=elements)


def _args_format(spec: ModelFormatSpec, parameters: dict[str, Any] | bool) -> Format:
    assert spec.tool_calls is not None
    encoding = spec.tool_calls.args_encoding
    if encoding == "qwen_xml":
        return QwenXMLParameterFormat(json_schema=_normalize_object_schema(parameters))
    if encoding == "json":
        return JSONSchemaFormat(json_schema=parameters)
    if encoding == "arg_key_value_xml":
        # any_order matches the parser's leniency and the renderer's
        # dict-insertion order, which need not match schema order.
        return JSONSchemaFormat(
            json_schema=_normalize_object_schema(parameters),
            style="glm_xml",
            any_order=True,
        )
    if encoding == "dsml":
        return JSONSchemaFormat(
            json_schema=_normalize_object_schema(parameters),
            style="deepseek_xml",
            any_order=True,
        )
    if encoding == "minimax_ns_xml":
        return _minimax_ns_args_format(parameters)
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
    shape = spec.tool_calls
    tags = tool_tag_formats(spec, tools)

    def calls_run(stop_after_first: bool = False) -> Format:
        return TagsWithSeparatorFormat(
            tags=tags,
            separator=shape.separator,
            at_least_one=True,
            stop_after_first=stop_after_first,
        )

    def wrap_section(calls: Format, with_prefix: bool = False) -> Format:
        if not shape.section_begin:
            return calls
        section: Format = TagFormat(
            begin=shape.section_begin, content=calls, end=shape.section_end
        )
        if with_prefix and shape.section_prefix:
            section = SequenceFormat(
                elements=[
                    ConstStringFormat(value=shape.section_prefix),
                    section,
                ]
            )
        return section

    suffix: Format
    if tool_choice == "auto":
        if not tags:
            suffix = AnyTextFormat()
        elif shape.section_begin:
            suffix = TriggeredTagsFormat(
                triggers=[shape.trigger], tags=[wrap_section(calls_run())]
            )
        else:
            suffix = TriggeredTagsFormat(triggers=[shape.trigger], tags=tags)
    elif tool_choice == "forced":
        suffix = wrap_section(calls_run(stop_after_first=True), with_prefix=True)
    elif tool_choice == "required":
        suffix = wrap_section(calls_run(), with_prefix=True)
    else:
        raise ValueError(f"unsupported tool_choice: {tool_choice!r}")

    return StructuralTag(format=suffix)
