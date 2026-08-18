# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generality check: recent model families driven from one spec each.

Where an in-tree engine config exists (GLM-4.7/5.x, DeepSeek-V4) the
generated parser must match it behaviorally; Step-3.5 is compared against
its legacy hand-written parser; MiniMax-M3 (Rust-only upstream) is pinned
to verbatim wire fixtures. Every family also closes the renderer → parser
→ grammar loop.
"""

import json
from unittest.mock import MagicMock

import jinja2
import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.format_spec.canonicalize import canonicalize_structured_outputs
from vllm.format_spec.parser_gen import to_parser_engine_config
from vllm.format_spec.specs import (
    DEEPSEEK_V4_SPEC,
    GLM_SPEC,
    MINIMAX_M3_SPEC,
    QWEN35_SPEC,
    STEP3P5_SPEC,
)
from vllm.format_spec.tag_gen import tool_structural_tag
from vllm.format_spec.template_gen import (
    render_assistant_turn,
    to_reference_template,
)
from vllm.format_spec.validate import tag_accepts
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.sampling_params import StructuredOutputsParams

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


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


def extract_all(parser: ParserEngine, wire: str):
    req = make_request()
    reasoning = parser.extract_reasoning(wire, req)
    result = parser.extract_tool_calls(wire, req)
    calls = [
        (c.function.name, json.loads(c.function.arguments)) for c in result.tool_calls
    ]
    return reasoning, result.tools_called, calls, result.content


class TestGlmParity:
    """Generated GLM config vs the hand-written glm47_moe_config."""

    FIXTURES = [
        "plain answer",
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
        "<arg_value>Beijing</arg_value>\n</tool_call>",
        "<tool_call>get_time\n</tool_call>",
        "<tool_call>get_current_date</tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>",
        "<tool_call>w\n<arg_key>city</arg_key><arg_value>Dallas</arg_value>\n"
        "</tool_call>\n<tool_call>w\n<arg_key>city</arg_key>"
        "<arg_value>Orlando</arg_value>\n</tool_call>",
        "pondering</think>Let me check. <tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>",
    ]

    @pytest.mark.parametrize("wire", FIXTURES)
    def test_equivalence(self, wire):
        from vllm.parser.glm47_moe import glm47_moe_config

        vocab = {
            "<think>": 50,
            "</think>": 51,
            "<tool_call>": 60,
            "</tool_call>": 61,
        }
        reference = ParserEngine(
            make_mock_tokenizer(vocab),
            parser_engine_config=glm47_moe_config(thinking=True),
        )
        assert extract_all(make_parser(GLM_SPEC), wire) == extract_all(reference, wire)

    def test_grammar_accepts_renderer(self):
        wire = render_assistant_turn(
            GLM_SPEC, tool_calls=[("get_weather", {"city": "Beijing"})]
        )
        for tool_choice in ("auto", "required", "forced"):
            tag = tool_structural_tag(
                GLM_SPEC, [("get_weather", CITY_SCHEMA)], tool_choice
            )
            assert tag_accepts(tag, wire), (tool_choice, wire)

    def test_grammar_rejects_unknown_key(self):
        tag = tool_structural_tag(GLM_SPEC, [("get_weather", CITY_SCHEMA)], "required")
        wire = render_assistant_turn(
            GLM_SPEC, tool_calls=[("get_weather", {"town": "Beijing"})]
        )
        assert not tag_accepts(tag, wire)

    def test_reasoning_canonicalization(self):
        tag = canonicalize_structured_outputs(
            GLM_SPEC, StructuredOutputsParams(json=CITY_SCHEMA)
        )
        # <think> is prompt-seeded: output has only </think>.
        assert tag_accepts(tag, 'let me think</think>{"city": "Beijing"}')
        assert not tag_accepts(tag, '<think>x</think>{"city": "Beijing"}')

    def test_streaming_with_tools_keeps_calls(self):
        from tests.parser.engine.streaming_helpers import simulate_tool_streaming
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={"name": "get_weather", "parameters": CITY_SCHEMA},
            )
        ]
        config = to_parser_engine_config(GLM_SPEC)
        vocab = {
            text: 100 + i for i, text in enumerate(config.token_id_terminals.values())
        }
        parser = ParserEngine(
            make_mock_tokenizer(vocab), tools=tools, parser_engine_config=config
        )
        wire = (
            "<tool_call>get_weather\n<arg_key>city</arg_key>"
            "<arg_value>Beijing</arg_value>\n</tool_call>"
        )
        chunks = [wire[i : i + 4] for i in range(0, len(wire), 4)]
        results = simulate_tool_streaming(parser, make_request(), chunks)
        names = [
            tc.function.name
            for dm, _ in results
            if dm and dm.tool_calls
            for tc in dm.tool_calls
            if tc.function and tc.function.name
        ]
        assert names == ["get_weather"]


class TestDeepSeekV4Parity:
    """Generated DSML config vs the hand-written deepseek_v4_config."""

    FIXTURES = [
        "plain answer",
        "The user wants weather.</think>\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>'
        "\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="calc">\n'
        '<｜DSML｜parameter name="a" string="false">42</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="note" string="true">He said "hi"</｜DSML｜parameter>'
        "\n</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>'
        "\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    ]

    @pytest.mark.parametrize("wire", FIXTURES)
    def test_equivalence(self, wire):
        from vllm.parser.deepseek_v4 import deepseek_v4_config

        vocab = {"<think>": 50, "</think>": 51}
        reference = ParserEngine(
            make_mock_tokenizer(vocab),
            parser_engine_config=deepseek_v4_config(thinking=True),
        )
        assert extract_all(make_parser(DEEPSEEK_V4_SPEC), wire) == extract_all(
            reference, wire
        )

    def test_non_string_values_typed(self):
        wire = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="calc">\n'
            '<｜DSML｜parameter name="a" string="false">42</｜DSML｜parameter>'
            "\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
        _, _, calls, _ = extract_all(make_parser(DEEPSEEK_V4_SPEC), wire)
        assert calls == [("calc", {"a": 42})]

    def test_grammar_accepts_renderer(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["a", "note"],
            "additionalProperties": False,
        }
        wire = render_assistant_turn(
            DEEPSEEK_V4_SPEC, tool_calls=[("calc", {"a": 42, "note": "x y"})]
        )
        for tool_choice in ("required", "forced"):
            tag = tool_structural_tag(DEEPSEEK_V4_SPEC, [("calc", schema)], tool_choice)
            assert tag_accepts(tag, wire), (tool_choice, wire)

    def test_thinking_disabled_passthrough(self):
        # Non-thinking DSV4 prefills </think> in the prompt: output has no
        # reasoning markers at all.
        tag = canonicalize_structured_outputs(
            DEEPSEEK_V4_SPEC, StructuredOutputsParams(json=CITY_SCHEMA), thinking=False
        )
        assert tag_accepts(tag, '{"city": "Beijing"}')
        assert not tag_accepts(tag, 'x</think>{"city": "Beijing"}')


class TestStep3p5:
    """Step-3.5 reuses the qwen_xml encoding; parity vs the legacy parser."""

    WIRE = (
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n</parameter>\n"
        "<parameter=state>\nTX\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    def test_tool_calls_match_legacy(self):
        from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

        legacy = Step3p5ToolParser(MagicMock())
        want = legacy.extract_tool_calls(self.WIRE, make_request())
        _, tools_called, calls, _ = extract_all(make_parser(STEP3P5_SPEC), self.WIRE)
        assert tools_called == want.tools_called
        assert calls == [
            (c.function.name, json.loads(c.function.arguments)) for c in want.tool_calls
        ]

    def test_forced_reasoning_wire(self):
        parser = make_parser(STEP3P5_SPEC, thinking=False)  # forced overrides
        reasoning, rest = parser.extract_reasoning(
            "planning</think>done", make_request()
        )
        assert reasoning == "planning"
        assert rest == "done"


class TestMinimaxM3:
    """MiniMax-M3: pinned to verbatim wire fixtures (upstream parser is Rust)."""

    NS = "]<]minimax[>["
    WIRE = (
        ']<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name="get_weather">'
        "]<]minimax[>[<city>Seattle]<]minimax[>[</city>]<]minimax[>[</invoke>"
        "\n]<]minimax[>[</tool_call>"
    )

    def test_renderer_reproduces_fixture(self):
        wire = render_assistant_turn(
            MINIMAX_M3_SPEC, tool_calls=[("get_weather", {"city": "Seattle"})]
        )
        assert wire == self.WIRE

    def test_parser_round_trip(self):
        _, tools_called, calls, content = extract_all(
            make_parser(MINIMAX_M3_SPEC), self.WIRE
        )
        assert tools_called
        assert calls == [("get_weather", {"city": "Seattle"})]
        assert not (content or "").strip()

    def test_nested_args_round_trip(self):
        args = {"shipping": {"city": "Singapore", "zip": "018956"}}
        wire = render_assistant_turn(MINIMAX_M3_SPEC, tool_calls=[("order", args)])
        _, _, calls, _ = extract_all(make_parser(MINIMAX_M3_SPEC), wire)
        assert calls == [("order", args)]

    def test_grammar_accepts_renderer(self):
        wire = render_assistant_turn(
            MINIMAX_M3_SPEC, tool_calls=[("get_weather", {"city": "Seattle"})]
        )
        tag = tool_structural_tag(
            MINIMAX_M3_SPEC, [("get_weather", CITY_SCHEMA)], "required"
        )
        assert tag_accepts(tag, wire), wire

    def test_reasoning_markers(self):
        parser = make_parser(MINIMAX_M3_SPEC)
        reasoning, rest = parser.extract_reasoning(
            "pondering</mm:think>answer", make_request()
        )
        assert reasoning == "pondering"
        assert rest == "answer"

    def test_item_key_and_empty_collections_rejected(self):
        with pytest.raises(ValueError, match="item"):
            render_assistant_turn(MINIMAX_M3_SPEC, tool_calls=[("f", {"item": "solo"})])
        with pytest.raises(ValueError, match="empty"):
            render_assistant_turn(MINIMAX_M3_SPEC, tool_calls=[("f", {"tags": []})])
        with pytest.raises(ValueError, match="item"):
            tool_structural_tag(
                MINIMAX_M3_SPEC,
                [("f", {"type": "object", "properties": {"item": {"type": "string"}}})],
                "required",
            )


class TestKimiK3:
    """K3 XTML: parser/renderer spec-generated, tag delegated to registry."""

    WIRE = (
        "<|open|>think<|sep|>step<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer<|close|>response<|sep|>"
        '<|open|>tools<|sep|><|open|>call tool="calc" index="1"<|sep|>'
        '<|open|>argument key="x" type="number"<|sep|>1<|close|>argument<|sep|>'
        "<|close|>call<|sep|><|close|>tools<|sep|>"
    )

    def test_renderer_reproduces_fixture(self):
        from vllm.format_spec.specs import KIMI_K3_SPEC

        wire = render_assistant_turn(
            KIMI_K3_SPEC,
            reasoning="step",
            content="answer",
            tool_calls=[("calc", {"x": 1})],
        )
        assert wire == self.WIRE

    def test_parser_round_trip(self):
        from vllm.format_spec.specs import KIMI_K3_SPEC

        parser = make_parser(KIMI_K3_SPEC)
        reasoning, rest = parser.extract_reasoning(self.WIRE, make_request())
        assert reasoning == "step"
        _, tools_called, calls, content = extract_all(
            make_parser(KIMI_K3_SPEC), self.WIRE
        )
        assert tools_called
        assert calls == [("calc", {"x": 1})]
        assert content == "answer"

    def test_parity_with_legacy_parser(self):
        from vllm.format_spec.specs import KIMI_K3_SPEC
        from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        tokenizer.encode.side_effect = lambda text, **kw: [ord(c) for c in text]
        legacy = KimiK3ToolParser(tokenizer)
        want = legacy.extract_tool_calls(self.WIRE, make_request())
        _, tools_called, calls, _ = extract_all(make_parser(KIMI_K3_SPEC), self.WIRE)
        assert tools_called == want.tools_called
        assert calls == [
            (c.function.name, json.loads(c.function.arguments)) for c in want.tool_calls
        ]

    def test_attr_escaping_and_guards(self):
        from vllm.format_spec.specs import KIMI_K3_SPEC

        wire = render_assistant_turn(KIMI_K3_SPEC, tool_calls=[('na"me', {"x": "v"})])
        assert 'tool="na&quot;me"' in wire
        with pytest.raises(ValueError, match="unrenderable"):
            render_assistant_turn(
                KIMI_K3_SPEC,
                tool_calls=[("f", {"x": "bad<|close|>argument<|sep|>bytes"})],
            )


class TestReferenceTemplates:
    """Generated Jinja fragments stay byte-identical to the renderer."""

    CASES = [
        (GLM_SPEC, [("get_weather", {"city": "Beijing", "days": 3})]),
        (DEEPSEEK_V4_SPEC, [("calc", {"a": 42, "note": "x"})]),
        (MINIMAX_M3_SPEC, [("get_weather", {"city": "Seattle"})]),
        (STEP3P5_SPEC, [("get_weather", {"city": "Dallas"})]),
        (QWEN35_SPEC, [("get_weather", {"city": "Tokyo"})]),
    ]

    @pytest.mark.parametrize("spec,calls", CASES, ids=lambda p: getattr(p, "name", ""))
    def test_template_matches_renderer(self, spec, calls):
        template = jinja2.Environment().from_string(to_reference_template(spec))
        message = {
            "reasoning_content": "why not",
            "content": "sure",
            "tool_calls": [
                {"function": {"name": name, "arguments": args}} for name, args in calls
            ],
        }
        want = render_assistant_turn(
            spec, reasoning="why not", content="sure", tool_calls=calls
        )
        assert template.render(message=message) == want


class TestQwen35Variant:
    def test_start_in_prompt_reasoning(self):
        tag = canonicalize_structured_outputs(
            QWEN35_SPEC, StructuredOutputsParams(json=CITY_SCHEMA)
        )
        assert tag_accepts(tag, 'hmm</think>{"city": "Tokyo"}')
        assert not tag_accepts(tag, '<think>hmm</think>{"city": "Tokyo"}')
