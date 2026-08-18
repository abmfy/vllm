# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-artifact consistency checks for a :class:`ModelFormatSpec`."""

from __future__ import annotations

from xgrammar.structural_tag import StructuralTag


def tag_accepts(tag: StructuralTag, wire: str) -> bool:
    """Whether the compiled structural tag accepts ``wire`` to termination.

    Tokenizer-free: uses xgrammar's grammar-level matcher, so it checks
    byte-level agreement between the grammar and the wire format.
    """
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    grammar = xgr.Grammar.from_structural_tag(tag)
    return _is_grammar_accept_string(grammar, wire)
