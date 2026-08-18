# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pilot :class:`ModelFormatSpec` definitions.

Each spec is the ONLY place its model family's format literals appear;
parser, structural tags, constraint canonicalization, wire renderer, and
the reference template fragment are all generated from it.
"""

from vllm.format_spec.spec import ModelFormatSpec, ReasoningShape, ToolCallShape

QWEN3_SPEC = ModelFormatSpec(
    name="qwen3",
    reasoning=ReasoningShape(start="<think>", end="</think>"),
    tool_calls=ToolCallShape(
        trigger="<tool_call>",
        call_begin="<tool_call>\n",
        name_prefix="<function=",
        name_suffix=">\n",
        args_encoding="qwen_xml",
        call_end="\n</function>\n</tool_call>",
    ),
)

HERMES_SPEC = ModelFormatSpec(
    name="hermes",
    reasoning=None,
    tool_calls=ToolCallShape(
        trigger="<tool_call>",
        call_begin="<tool_call>\n",
        name_prefix='{"name": "',
        name_suffix='", "arguments": ',
        args_encoding="json",
        call_end="}\n</tool_call>",
    ),
)

DEEPSEEK_R1_SPEC = ModelFormatSpec(
    name="deepseek_r1",
    reasoning=ReasoningShape(
        start="<think>",
        end="</think>",
        start_in_prompt=True,
        forced=True,
        toggle_kwarg=None,
    ),
    tool_calls=None,
)

FORMAT_SPECS = {spec.name: spec for spec in (QWEN3_SPEC, HERMES_SPEC, DEEPSEEK_R1_SPEC)}
