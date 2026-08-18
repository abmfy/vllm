# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One :class:`ModelFormatSpec` must yield mutually consistent artifacts.

The tests here are the point of the experiment: the wire renderer, the
streaming/non-streaming parser, the tool-calling structural tag, the
reasoning-aware canonicalized constraint, and the reference Jinja template
are all generated from the same spec, so they must agree on the same bytes.
"""

import json
from unittest.mock import MagicMock

import jinja2
import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.format_spec.canonicalize import (
    canonicalize_structured_outputs,
    wrap_with_reasoning,
)
from vllm.format_spec.parser_gen import to_parser_engine_config
from vllm.format_spec.specs import DEEPSEEK_R1_SPEC, HERMES_SPEC, QWEN3_SPEC
from vllm.format_spec.tag_gen import tool_structural_tag
from vllm.format_spec.template_gen import (
    render_assistant_turn,
    to_reference_template,
)
from vllm.format_spec.validate import tag_accepts
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import qwen3_config
from vllm.sampling_params import StructuredOutputsParams

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
TOOLS = [("get_weather", WEATHER_SCHEMA)]


def make_request() -> MagicMock:
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def make_parser(spec, thinking: bool = True) -> ParserEngine:
    config = to_parser_engine_config(spec, thinking=thinking)
    vocab = {text: 100 + i for i, text in enumerate(config.token_id_terminals.values())}
    return ParserEngine(make_mock_tokenizer(vocab), parser_engine_config=config)


class TestGeneratedParser:
    """Generated parser configs must match the hand-written Qwen3 one."""

    QWEN3_FIXTURES = [
        "plain answer, no markup",
        "<think>pondering</think>the answer",
        (
            "<think>let me check</think>I'll look it up.\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nTokyo\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
    ]

    @pytest.mark.parametrize("wire", QWEN3_FIXTURES)
    def test_qwen3_equivalence_non_streaming(self, wire):
        generated = make_parser(QWEN3_SPEC)
        vocab = {"<tool_call>": 100, "</tool_call>": 101}
        reference = ParserEngine(
            make_mock_tokenizer(vocab), parser_engine_config=qwen3_config(thinking=True)
        )
        req = make_request()

        got_reasoning = generated.extract_reasoning(wire, req)
        want_reasoning = reference.extract_reasoning(wire, req)
        assert got_reasoning == want_reasoning

        got_tools = generated.extract_tool_calls(wire, req)
        want_tools = reference.extract_tool_calls(wire, req)
        assert got_tools.tools_called == want_tools.tools_called
        got_calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in got_tools.tool_calls
        ]
        want_calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in want_tools.tool_calls
        ]
        assert got_calls == want_calls
        assert got_tools.content == want_tools.content

    def test_hermes_parser_from_spec(self):
        parser = make_parser(HERMES_SPEC)
        wire = (
            "Checking.\n<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>"
        )
        result = parser.extract_tool_calls(wire, make_request())
        assert result.tools_called
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
        assert result.content == "Checking."

    def test_hermes_parallel_calls_and_wrapper_variance(self):
        parser = make_parser(HERMES_SPEC)
        # Second call omits the newline before </tool_call>; nested args.
        wire = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>\n<tool_call>\n"
            '{"name": "get_weather", "arguments": {"geo": {"lat": 1}}}'
            "</tool_call>"
        )
        result = parser.extract_tool_calls(wire, make_request())
        calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in result.tool_calls
        ]
        assert calls == [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"geo": {"lat": 1}}),
        ]
        # Separator whitespace between calls must not leak into content.
        assert not (result.content or "").strip()

    def test_hermes_streaming_args_complete(self):
        from tests.parser.engine.streaming_helpers import (
            collect_tool_arguments,
            simulate_tool_streaming,
        )

        parser = make_parser(HERMES_SPEC)
        wire = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>"
        )
        chunks = [wire[i : i + 7] for i in range(0, len(wire), 7)]
        results = simulate_tool_streaming(parser, make_request(), chunks)
        args = collect_tool_arguments(results)
        assert json.loads(args) == {"city": "Paris"}


class TestStructuralTag:
    """The generated grammar must accept exactly what the renderer emits."""

    @pytest.mark.parametrize("spec", [QWEN3_SPEC, HERMES_SPEC], ids=lambda s: s.name)
    def test_renderer_output_accepted(self, spec):
        wire = render_assistant_turn(
            spec, tool_calls=[("get_weather", {"city": "Paris"})]
        )
        for tool_choice in ("auto", "required", "forced"):
            tag = tool_structural_tag(spec, TOOLS, tool_choice)
            assert tag_accepts(tag, wire), (spec.name, tool_choice, wire)

    @pytest.mark.parametrize("spec", [QWEN3_SPEC, HERMES_SPEC], ids=lambda s: s.name)
    def test_free_text_only_valid_for_auto(self, spec):
        tag_auto = tool_structural_tag(spec, TOOLS, "auto")
        tag_required = tool_structural_tag(spec, TOOLS, "required")
        assert tag_accepts(tag_auto, "no tools needed here")
        assert not tag_accepts(tag_required, "no tools needed here")

    def test_schema_violation_rejected(self):
        tag = tool_structural_tag(HERMES_SPEC, TOOLS, "required")
        wire = render_assistant_turn(
            HERMES_SPEC, tool_calls=[("get_weather", {"town": "Paris"})]
        )
        assert not tag_accepts(tag, wire)

    def test_unknown_tool_name_rejected(self):
        tag = tool_structural_tag(HERMES_SPEC, TOOLS, "required")
        wire = render_assistant_turn(
            HERMES_SPEC, tool_calls=[("get_wather", {"city": "Paris"})]
        )
        assert not tag_accepts(tag, wire)


class TestCanonicalization:
    """Free-form reasoning + user constraint composed into one grammar."""

    def test_reasoning_then_json_schema(self):
        params = StructuredOutputsParams(json=WEATHER_SCHEMA)
        tag = canonicalize_structured_outputs(QWEN3_SPEC, params)
        assert tag is not None
        assert tag_accepts(tag, '<think>hmm</think>{"city": "Paris"}')
        # Reasoning is optional for a toggleable thinker.
        assert tag_accepts(tag, '{"city": "Paris"}')
        # The constrained part alone must still be enforced.
        assert not tag_accepts(tag, "<think>hmm</think>not json")
        assert not tag_accepts(tag, '<think>hmm</think>{"town": "Paris"}')
        # EOS inside reasoning is masked: reasoning alone can't terminate.
        assert not tag_accepts(tag, "<think>hmm</think>")

    def test_start_in_prompt_shape(self):
        params = StructuredOutputsParams(json=WEATHER_SCHEMA)
        tag = canonicalize_structured_outputs(DEEPSEEK_R1_SPEC, params)
        assert tag is not None
        # R1's <think> is pre-filled in the prompt: output has only </think>.
        assert tag_accepts(tag, 'reasoning here</think>{"city": "Paris"}')
        assert not tag_accepts(tag, '<think>x</think>{"city": "Paris"}')
        # Forced reasoning: the end marker must appear.
        assert not tag_accepts(tag, '{"city": "Paris"}')

    def test_thinking_disabled_passthrough(self):
        params = StructuredOutputsParams(json=WEATHER_SCHEMA)
        tag = canonicalize_structured_outputs(QWEN3_SPEC, params, thinking=False)
        assert tag is not None
        assert tag_accepts(tag, '{"city": "Paris"}')
        assert not tag_accepts(tag, '<think>hmm</think>{"city": "Paris"}')

    def test_forced_reasoning_ignores_thinking_flag(self):
        from vllm.format_spec.parser_gen import to_parser_engine_config
        from vllm.parser.engine.parser_engine_config import ParserState

        params = StructuredOutputsParams(json=WEATHER_SCHEMA)
        tag = canonicalize_structured_outputs(DEEPSEEK_R1_SPEC, params, thinking=False)
        # Grammar and parser must agree: R1 always starts inside reasoning.
        assert tag_accepts(tag, 'still thinking</think>{"city": "Paris"}')
        assert not tag_accepts(tag, '{"city": "Paris"}')
        config = to_parser_engine_config(DEEPSEEK_R1_SPEC, thinking=False)
        assert config.initial_state == ParserState.REASONING

    def test_end_only_optional_reasoning(self):
        from vllm.format_spec.spec import ModelFormatSpec, ReasoningShape

        spec = ModelFormatSpec(
            name="end_only",
            reasoning=ReasoningShape(start=None, end="</think>"),
        )
        params = StructuredOutputsParams(json=WEATHER_SCHEMA)
        tag = canonicalize_structured_outputs(spec, params)
        assert tag_accepts(tag, 'pondering</think>{"city": "Paris"}')
        assert tag_accepts(tag, '{"city": "Paris"}')

    def test_choice_and_regex(self):
        tag = canonicalize_structured_outputs(
            QWEN3_SPEC, StructuredOutputsParams(choice=["yes", "no"])
        )
        assert tag_accepts(tag, "<think>weighing</think>yes")
        assert not tag_accepts(tag, "<think>weighing</think>maybe")


class TestReferenceTemplate:
    """The generated Jinja fragment renders byte-identically to the renderer."""

    CASES = [
        {"reasoning": "thinking hard", "content": "the answer", "tool_calls": []},
        {"reasoning": None, "content": "plain", "tool_calls": []},
        {
            "reasoning": "need a tool",
            "content": "",
            "tool_calls": [("get_weather", {"city": "Paris"})],
        },
        {
            "reasoning": None,
            "content": "two calls",
            "tool_calls": [
                ("get_weather", {"city": "Paris"}),
                ("get_weather", {"city": "Tokyo"}),
            ],
        },
    ]

    @pytest.mark.parametrize("spec", [QWEN3_SPEC, HERMES_SPEC], ids=lambda s: s.name)
    @pytest.mark.parametrize("case", CASES)
    def test_template_matches_renderer(self, spec, case):
        if case["reasoning"] is not None and spec.reasoning is None:
            pytest.skip("spec has no reasoning section")
        template = jinja2.Environment().from_string(to_reference_template(spec))
        message = {
            "reasoning_content": case["reasoning"],
            "content": case["content"],
            "tool_calls": [
                {"function": {"name": name, "arguments": args}}
                for name, args in case["tool_calls"]
            ],
        }
        want = render_assistant_turn(
            spec,
            reasoning=case["reasoning"],
            content=case["content"],
            tool_calls=case["tool_calls"],
        )
        assert template.render(message=message) == want


class TestNonStringArgs:
    """Non-string argument values must stay consistent across artifacts."""

    ARGS = {"city": "Paris", "days": 3, "detailed": True}
    SCHEMA = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "detailed": {"type": "boolean"},
        },
        "required": ["city", "days", "detailed"],
        "additionalProperties": False,
    }

    @pytest.mark.parametrize("spec", [QWEN3_SPEC, HERMES_SPEC], ids=lambda s: s.name)
    def test_renderer_grammar_agreement(self, spec):
        wire = render_assistant_turn(spec, tool_calls=[("get_weather", self.ARGS)])
        tag = tool_structural_tag(spec, [("get_weather", self.SCHEMA)], "required")
        assert tag_accepts(tag, wire), wire

    def test_template_matches_renderer_non_string(self):
        template = jinja2.Environment().from_string(to_reference_template(QWEN3_SPEC))
        message = {
            "reasoning_content": None,
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": self.ARGS}}
            ],
        }
        want = render_assistant_turn(
            QWEN3_SPEC, tool_calls=[("get_weather", self.ARGS)]
        )
        assert template.render(message=message) == want

    def test_unrenderable_qwen_xml_value_raises(self):
        with pytest.raises(ValueError, match="unrenderable"):
            render_assistant_turn(
                QWEN3_SPEC, tool_calls=[("f", {"x": "a</parameter>b"})]
            )

    def test_content_none(self):
        wire = render_assistant_turn(HERMES_SPEC, content=None, tool_calls=[("f", {})])
        assert wire.startswith("<tool_call>")


class TestThreeWayConsistency:
    """Renderer bytes are parsed back losslessly AND accepted by the grammar."""

    def test_qwen3_full_loop(self):
        reasoning, content = "check the weather", ""
        calls = [("get_weather", {"city": "Paris"})]
        wire = render_assistant_turn(
            QWEN3_SPEC, reasoning=reasoning, content=content, tool_calls=calls
        )

        parser = make_parser(QWEN3_SPEC)
        req = make_request()
        got_reasoning, _rest = parser.extract_reasoning(wire, req)
        assert got_reasoning == reasoning
        result = make_parser(QWEN3_SPEC).extract_tool_calls(wire, req)
        assert [
            (c.function.name, json.loads(c.function.arguments))
            for c in result.tool_calls
        ] == calls

        tool_tag = tool_structural_tag(QWEN3_SPEC, TOOLS, "auto")
        full_tag = wrap_with_reasoning(QWEN3_SPEC, tool_tag.format)
        assert tag_accepts(full_tag, wire)
