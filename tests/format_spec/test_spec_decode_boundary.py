# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-decode validation: the reasoning boundary is a non-event.

With a canonicalized whole-output tag, a draft window may end at ANY byte
position — before, on, or after the reasoning-end marker — and the prefix
is always grammar-viable. This is the property that makes the scheduler's
mid-draft boundary machinery (simulated windows, trim-then-advance,
tolerated rejections) unnecessary: draft validation and bitmask filling
advance one grammar uniformly, with exact rollback.
"""

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from vllm.format_spec.canonicalize import canonicalize_structured_outputs
from vllm.format_spec.specs import DEEPSEEK_V4_SPEC, GLM_SPEC, QWEN3_SPEC
from vllm.sampling_params import StructuredOutputsParams

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

WIRES = {
    QWEN3_SPEC.name: '<think>let me see</think>{"city": "Paris"}',
    GLM_SPEC.name: 'let me see</think>{"city": "Paris"}',
    DEEPSEEK_V4_SPEC.name: 'let me see</think>{"city": "Paris"}',
}


@pytest.mark.parametrize(
    "spec", [QWEN3_SPEC, GLM_SPEC, DEEPSEEK_V4_SPEC], ids=lambda s: s.name
)
def test_every_prefix_is_viable(spec):
    """Any draft-window cut point is valid; only the full wire terminates."""
    tag = canonicalize_structured_outputs(spec, StructuredOutputsParams(json=SCHEMA))
    grammar = xgr.Grammar.from_structural_tag(tag)
    wire = WIRES[spec.name]
    for cut in range(len(wire) + 1):
        assert _is_grammar_accept_string(
            grammar, wire[:cut], require_termination=False
        ), (spec.name, cut, wire[:cut])
    # EOS is masked until the constrained suffix completes: no prefix that
    # stops inside reasoning (or mid-JSON) may terminate.
    for cut in range(len(wire)):
        assert not _is_grammar_accept_string(grammar, wire[:cut]), (spec.name, cut)
    assert _is_grammar_accept_string(grammar, wire)
