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
    # A start marker appearing mid-content re-enters reasoning (GLM, DSML).
    content_reenters_reasoning: bool = False
    # Whether marker terminals stay active when thinking is disabled
    # (GLM's reference config drops them; DeepSeek-V4's keeps them).
    markers_when_disabled: bool = True

    @property
    def emits_start(self) -> bool:
        """Whether ``start`` appears in the model's output stream."""
        return self.start is not None and not self.start_in_prompt


ArgsEncoding = Literal[
    "json", "qwen_xml", "arg_key_value_xml", "dsml", "minimax_ns_xml", "k3_xtml"
]

ARGS_ENCODINGS = (
    "json",
    "qwen_xml",
    "arg_key_value_xml",
    "dsml",
    "minimax_ns_xml",
    "k3_xtml",
)


@dataclass(frozen=True)
class ToolCallShape:
    """Wire format of a single tool call.

    A call renders as::

        {call_begin}{name_prefix}{NAME}{name_suffix}{ARGS}{call_end}

    with consecutive calls joined by ``separator`` and, when
    ``section_begin`` is set, the whole run wrapped in
    ``{section_begin}...{section_end}``. ARGS is encoded per
    ``args_encoding``:

    * ``"json"``: a JSON object matching the tool's parameter schema.
    * ``"qwen_xml"``: ``<parameter=KEY>\\nVALUE\\n</parameter>`` blocks
      (Qwen3-Coder/3.5+, Step-3.5).
    * ``"arg_key_value_xml"``: ``<arg_key>K</arg_key>\\n<arg_value>V
      </arg_value>\\n`` pairs (GLM-4.x/5.x, Ling3).
    * ``"dsml"``: ``<｜DSML｜parameter name="K" string="true|false">V
      </｜DSML｜parameter>`` lines (DeepSeek-V4 Flash/Pro).
    * ``"minimax_ns_xml"``: namespace-prefixed recursive element tags
      ``{NS}<K>V{NS}</K>`` (MiniMax-M3).

    Attributes:
        trigger: Literal that unambiguously starts the tool-call region;
            used both as the streaming lexer terminal and the
            structural-tag trigger.
        call_begin: Bytes opening one call.
        name_prefix: Bytes immediately preceding the function name.
        name_suffix: Bytes separating the name from the arguments.
        args_encoding: How arguments are encoded.
        call_end: Bytes closing one call.
        separator: Bytes between consecutive parallel calls.
        section_begin: Bytes opening the whole tool-call section, empty
            when calls are not section-wrapped.
        section_end: Bytes closing the section.
    """

    trigger: str
    call_begin: str
    name_prefix: str
    name_suffix: str
    args_encoding: ArgsEncoding
    call_end: str
    separator: str = "\n"
    section_begin: str = ""
    section_end: str = ""
    # Bytes emitted before section_begin (e.g. DSML's "\n\n"); when empty,
    # the renderer falls back to `separator` after non-empty content.
    section_prefix: str = ""
    # Per-call attribute bytes between name and name_suffix; "{index}" is
    # replaced with the 1-based call position (K3's index="N").
    call_attrs: str = ""

    def __post_init__(self) -> None:
        anchor = self.section_begin or self.call_begin
        if not anchor.startswith(self.trigger):
            raise ValueError(
                f"section_begin/call_begin {anchor!r} must start with "
                f"trigger {self.trigger!r}"
            )
        if self.args_encoding not in ARGS_ENCODINGS:
            raise ValueError(f"unsupported args_encoding: {self.args_encoding!r}")
        if bool(self.section_begin) != bool(self.section_end):
            raise ValueError("section_begin and section_end must be set together")


@dataclass(frozen=True)
class ModelFormatSpec:
    """Complete declarative output-format description of a model family."""

    name: str
    reasoning: ReasoningShape | None = None
    tool_calls: ToolCallShape | None = None
    # (open, close) markers wrapping plain content (K3's response channel).
    content_wrapper: tuple[str, str] | None = None
    # Trailing markers the model may emit before EOS (dropped by the parser).
    turn_end_markers: tuple[str, ...] = ()
