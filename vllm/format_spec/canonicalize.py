# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-aware canonicalization of user constraints.

Generalizes the harmony-only ``_adjust_output_format`` in
``vllm/parser/harmony.py``: any user constraint (JSON schema, regex,
choice, grammar, structural tag) is rewritten into a structural tag that
covers the WHOLE output — free-form reasoning first, then the constrained
final content. The grammar matcher then owns the reasoning boundary, so
requests canonicalized this way no longer need the scheduler-side gating
machinery (``is_reasoning_end_streaming``, mid-draft boundary handling).
Wiring status and migration steps: ``docs/design/unified_format_spec.md``.
"""

from __future__ import annotations

import json

from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    GrammarFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)

from vllm.format_spec.spec import ModelFormatSpec
from vllm.sampling_params import StructuredOutputsParams

_JSON_OBJECT = JSONSchemaFormat(json_schema={"type": "object"})


def params_to_format(params: StructuredOutputsParams) -> Format | None:
    """Map user structured-output params onto an xgrammar Format."""
    if params.json_object:
        return _JSON_OBJECT
    if params.json is not None:
        schema = params.json
        if isinstance(schema, str):
            schema = json.loads(schema)
        return JSONSchemaFormat(json_schema=schema)
    if params.regex is not None:
        return RegexFormat(pattern=params.regex)
    if params.choice is not None:
        return OrFormat(
            elements=[ConstStringFormat(value=choice) for choice in params.choice]
        )
    if params.grammar is not None:
        return GrammarFormat(grammar=params.grammar)
    if params.structural_tag is not None:
        s_tag = json.loads(params.structural_tag)
        if "structures" in s_tag:
            raise ValueError(
                "legacy structural_tag cannot be reasoning-wrapped; "
                "use the format-based spec"
            )
        return StructuralTag.model_validate(s_tag).format
    return None


def reasoning_element(spec: ModelFormatSpec) -> Format | None:
    """The reasoning-section Format preceding the constrained content.

    Returns None when the spec has no reasoning section. The section body
    excludes both markers so a fake boundary cannot appear inside it.
    """
    shape = spec.reasoning
    if shape is None:
        return None
    excludes = [shape.end]
    if shape.start is not None:
        excludes.append(shape.start)
    body = AnyTextFormat(excludes=excludes)
    begin = shape.start if shape.emits_start else ""
    section: Format = TagFormat(begin=begin or "", content=body, end=shape.end)
    if not shape.forced and not shape.start_in_prompt:
        # The model may answer directly without a reasoning section.
        section = OptionalFormat(content=section)
    return section


def wrap_with_reasoning(
    spec: ModelFormatSpec,
    final: Format,
    thinking: bool = True,
) -> StructuralTag:
    """Compose free-form reasoning ahead of the constrained final content.

    The resulting tag constrains the full output from token 0: EOS is
    masked until the final content completes, and the reasoning boundary
    is tracked by the grammar matcher itself — no scheduler-side gating.
    """
    if spec.reasoning is not None and spec.reasoning.forced:
        thinking = True
    element = reasoning_element(spec) if thinking else None
    if element is None:
        return StructuralTag(format=final)
    return StructuralTag(format=SequenceFormat(elements=[element, final]))


def canonicalize_structured_outputs(
    spec: ModelFormatSpec,
    params: StructuredOutputsParams,
    thinking: bool = True,
) -> StructuralTag | None:
    """Rewrite user constraints into a whole-output structural tag.

    Returns None when ``params`` carries no constraint.
    """
    final = params_to_format(params)
    if final is None:
        return None
    return wrap_with_reasoning(spec, final, thinking=thinking)
