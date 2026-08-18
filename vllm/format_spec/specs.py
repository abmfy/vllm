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

# GLM-4.5/4.6/4.7/5.x and Ling3 (xgrammar's glm_4_7 builtin covers GLM-5).
# <think> is seeded by the chat template; output carries only </think>.
GLM_SPEC = ModelFormatSpec(
    name="glm_4_7",
    reasoning=ReasoningShape(
        start="<think>",
        end="</think>",
        start_in_prompt=True,
        content_reenters_reasoning=True,
        markers_when_disabled=False,
    ),
    tool_calls=ToolCallShape(
        trigger="<tool_call>",
        call_begin="<tool_call>",
        name_prefix="",
        name_suffix="\n",
        args_encoding="arg_key_value_xml",
        call_end="</tool_call>",
    ),
)

# DeepSeek-V4-Flash and DeepSeek-V4-Pro (shared DSML wire format).
DEEPSEEK_V4_SPEC = ModelFormatSpec(
    name="deepseek_v4",
    reasoning=ReasoningShape(
        start="<think>",
        end="</think>",
        start_in_prompt=True,
        toggle_kwarg="thinking",
        content_reenters_reasoning=True,
    ),
    tool_calls=ToolCallShape(
        trigger="<｜DSML｜tool_calls>",
        section_begin="<｜DSML｜tool_calls>\n",
        section_end="\n</｜DSML｜tool_calls>",
        section_prefix="\n\n",
        call_begin="",
        name_prefix='<｜DSML｜invoke name="',
        name_suffix='">\n',
        args_encoding="dsml",
        call_end="\n</｜DSML｜invoke>",
    ),
)

_NS = "]<]minimax[>["

# MiniMax-M3 (namespace-prefixed XML; differs from M2's <minimax:tool_call>).
MINIMAX_M3_SPEC = ModelFormatSpec(
    name="minimax_m3",
    reasoning=ReasoningShape(
        start="<mm:think>",
        end="</mm:think>",
        start_in_prompt=True,
        toggle_kwarg="thinking_mode",
    ),
    tool_calls=ToolCallShape(
        trigger=f"{_NS}<tool_call>",
        section_begin=f"{_NS}<tool_call>\n",
        section_end=f"\n{_NS}</tool_call>",
        call_begin="",
        name_prefix=f'{_NS}<invoke name="',
        name_suffix='">',
        args_encoding="minimax_ns_xml",
        call_end=f"{_NS}</invoke>",
    ),
)

# Step-3.5 (stepfun-ai): tool wire is byte-identical to Qwen3-Coder XML;
# only the reasoning shape differs (always-on, <think> seeded in prompt).
STEP3P5_SPEC = ModelFormatSpec(
    name="step3p5",
    reasoning=ReasoningShape(
        start="<think>",
        end="</think>",
        start_in_prompt=True,
        forced=True,
        toggle_kwarg=None,
    ),
    tool_calls=QWEN3_SPEC.tool_calls,
)

# Qwen3.5/3.6/3.8: same tool wire as qwen3; template seeds <think> so the
# output stream carries only </think>.
QWEN35_SPEC = ModelFormatSpec(
    name="qwen3_5",
    reasoning=ReasoningShape(start="<think>", end="</think>", start_in_prompt=True),
    tool_calls=QWEN3_SPEC.tool_calls,
)

FORMAT_SPECS = {
    spec.name: spec
    for spec in (
        QWEN3_SPEC,
        HERMES_SPEC,
        DEEPSEEK_R1_SPEC,
        GLM_SPEC,
        DEEPSEEK_V4_SPEC,
        MINIMAX_M3_SPEC,
        STEP3P5_SPEC,
        QWEN35_SPEC,
    )
}
