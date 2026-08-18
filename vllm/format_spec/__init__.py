# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.format_spec.canonicalize import (
    canonicalize_structured_outputs,
    wrap_with_reasoning,
)
from vllm.format_spec.parser_gen import to_parser_engine_config
from vllm.format_spec.spec import ModelFormatSpec, ReasoningShape, ToolCallShape
from vllm.format_spec.specs import FORMAT_SPECS
from vllm.format_spec.tag_gen import tool_structural_tag
from vllm.format_spec.template_gen import (
    render_assistant_turn,
    to_reference_template,
)

__all__ = [
    "FORMAT_SPECS",
    "ModelFormatSpec",
    "ReasoningShape",
    "ToolCallShape",
    "canonicalize_structured_outputs",
    "render_assistant_turn",
    "to_parser_engine_config",
    "to_reference_template",
    "tool_structural_tag",
    "wrap_with_reasoning",
]
