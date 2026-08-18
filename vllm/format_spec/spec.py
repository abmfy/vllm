# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-source-of-truth description of a model family's output format.

A :class:`ModelFormatSpec` declares the wire format a model uses for its
reasoning section and tool calls. Downstream artifacts that today hardcode
this knowledge independently are generated from the spec instead:

* a :class:`~vllm.parser.engine.parser_engine_config.ParserEngineConfig`
  (streaming reasoning + tool parsing) — see ``parser_gen``,
* xgrammar structural tags for constrained tool calling — see ``tag_gen``,
* a reasoning-aware canonicalization of arbitrary user constraints
  (generalizing the harmony ``_adjust_output_format``) — see ``canonicalize``,
* a reference chat-template fragment and a wire renderer used for
  round-trip validation — see ``template_gen``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReasoningShape:
    """Shape of the model's reasoning section.

    Attributes:
        start: Marker the model emits to open reasoning, or None when the
            model never emits it in its output stream.
        end: Marker that closes the reasoning section.
        start_in_prompt: True when the chat template appends ``start`` to
            the prompt, so generation begins already inside reasoning and
            the output stream never contains ``start``.
        forced: True when the model always reasons (no request-level toggle
            can disable it).
        toggle_kwarg: ``chat_template_kwargs`` key that toggles reasoning.
    """

    start: str | None
    end: str
    start_in_prompt: bool = False
    forced: bool = False
    toggle_kwarg: str | None = "enable_thinking"

    @property
    def emits_start(self) -> bool:
        """Whether ``start`` appears in the model's output stream."""
        return self.start is not None and not self.start_in_prompt


ArgsEncoding = Literal["json", "qwen_xml"]


@dataclass(frozen=True)
class ToolCallShape:
    """Wire format of a single tool call.

    A call renders as::

        {call_begin}{name_prefix}{NAME}{name_suffix}{ARGS}{call_end}

    with ARGS encoded per ``args_encoding``:

    * ``"json"``: a JSON object matching the tool's parameter schema.
    * ``"qwen_xml"``: ``<parameter=KEY>\\nVALUE\\n</parameter>`` blocks.

    Attributes:
        trigger: Literal that unambiguously starts a tool call; used both
            as the streaming lexer terminal and the structural-tag trigger.
        call_begin: Bytes opening one call (starts with ``trigger``).
        name_prefix: Bytes immediately preceding the function name.
        name_suffix: Bytes separating the name from the arguments.
        args_encoding: How arguments are encoded.
        call_end: Bytes closing one call.
        separator: Bytes between consecutive parallel calls.
    """

    trigger: str
    call_begin: str
    name_prefix: str
    name_suffix: str
    args_encoding: ArgsEncoding
    call_end: str
    separator: str = "\n"

    def __post_init__(self) -> None:
        if not self.call_begin.startswith(self.trigger):
            raise ValueError(
                f"call_begin {self.call_begin!r} must start with "
                f"trigger {self.trigger!r}"
            )
        if self.args_encoding not in ("json", "qwen_xml"):
            raise ValueError(f"unsupported args_encoding: {self.args_encoding!r}")


@dataclass(frozen=True)
class ModelFormatSpec:
    """Complete declarative output-format description of a model family."""

    name: str
    reasoning: ReasoningShape | None = None
    tool_calls: ToolCallShape | None = None
