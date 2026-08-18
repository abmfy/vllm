# Unified Model Format Spec (experiment)

Status: prototype on branch `bowen/unified-format-spec`. Package:
[`vllm/format_spec/`](../../vllm/format_spec/), tests:
[`tests/format_spec/`](../../tests/format_spec/).

## Problem

A model family's output format — reasoning markers, tool-call wrappers,
argument encoding — is currently re-encoded by hand in up to four places
that are paired only by convention:

1. the chat template (Jinja, usually shipped in the model repo),
2. a tool parser (`vllm/tool_parsers/`, or a `ParserEngineConfig`),
3. a reasoning parser (`vllm/reasoning/`),
4. structural-tag builders (`vllm/tool_parsers/structural_tag_registry.py`
   or xgrammar's builtin model tags).

`<think>`/`</think>` alone is independently hardcoded in three layers of
vLLM today. Nothing verifies the copies agree; every disagreement surfaces
as a runtime bug (wrong boundary detection, grammar/parser mismatch,
template/parser drift). The `vllm/parser` engine migration already merged
(2) and (3) for ~11 families; this experiment goes the rest of the way.

## Design

One frozen dataclass per model family, `ModelFormatSpec`
([spec.py](../../vllm/format_spec/spec.py)), declares only *literals and
shape flags*:

```python
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
```

Five artifacts are generated from it:

| Artifact | Generator | Replaces |
| --- | --- | --- |
| `ParserEngineConfig` (→ reasoning + tool parser via `make_adapters`) | [parser_gen.py](../../vllm/format_spec/parser_gen.py) | per-model parser classes |
| tool-calling structural tag | [tag_gen.py](../../vllm/format_spec/tag_gen.py) | per-model builders in `structural_tag_registry.py` |
| whole-output constraint (reasoning + user schema) | [canonicalize.py](../../vllm/format_spec/canonicalize.py) | harmony-only `_adjust_output_format` + the scheduler's reasoning-gating machinery |
| wire renderer + reference Jinja fragment | [template_gen.py](../../vllm/format_spec/template_gen.py) | by-convention template↔parser pairing |
| byte-level grammar acceptance checks | [validate.py](../../vllm/format_spec/validate.py) | (new) |

The FSM *topology* per argument encoding (`json`, `qwen_xml`) lives once in
`parser_gen.py`; specs contribute only strings. Adding a model family in
this scheme is one spec literal plus, when its argument encoding is new,
one topology function.

### Whole-output constraints kill the boundary machinery

`canonicalize.wrap_with_reasoning` composes

```text
sequence[ optional(tag(<think>, any_text ex {</think>}, </think>)), <user constraint> ]
```

so the grammar matcher itself owns the reasoning boundary from token 0.
Consequences, in decreasing order of importance:

* The scheduler-side gating (`is_reasoning_end_streaming`, mid-draft
  boundary windows, trim-then-advance — the #44297/#44993/#48200 problem
  space) becomes unnecessary for spec'd models: draft tokens advance one
  grammar; rollback is exact; stateful text-space parsers are out of the
  hot path entirely. (SGLang reaches the same property differently, via a
  rollback-able token-space wrapper FSM; harmony reaches it exactly this
  way for gpt-oss.)
* EOS is masked while reasoning is open, so "model EOS'd inside
  reasoning and the schema was silently violated" cannot happen.
* `ReasoningShape.start_in_prompt` / `forced` / `start=None` cover the
  DeepSeek-R1 (pre-filled `<think>`) and MiniMax-M2 (end-marker-only)
  shapes declaratively.

### What is deliberately NOT generated

The actual chat template stays externally owned (model repos ship it).
Instead the spec generates (a) the authoritative assistant-turn *wire
renderer* and (b) a reference Jinja fragment proven byte-identical to it,
and the test suite closes the loop: renderer output must round-trip
through the generated parser and be accepted by the compiled structural
tag (`xgrammar.Grammar.from_structural_tag`, tokenizer-free). Validating a
model repo's template against the spec (render a synthetic turn, compare)
is the natural follow-up and turns template↔parser pairing from
convention into CI.

## Family coverage

| Family (released models) | Encoding | Section | Parity reference | Status |
| --- | --- | --- | --- | --- |
| Qwen3-Coder/3.5/3.6/3.8, MiMo | `qwen_xml` | no | `qwen3_config` | equivalent |
| Step-3.5 (stepfun) | `qwen_xml` (reused) | no | `Step3p5ToolParser` | equivalent (tool wire) |
| Hermes / classic Qwen3 / QwQ | `json` | no | legacy hermes parser | canonical wire only |
| GLM-4.5/4.6/4.7/**5.x**, Ling3 | `arg_key_value_xml` | no | `glm47_moe_config` | equivalent |
| DeepSeek-V4 **Flash + Pro** | `dsml` | yes | `deepseek_v4_config` | equivalent |
| MiniMax-M3 | `minimax_ns_xml` (recursive) | yes | Rust-parser fixtures | fixture-pinned |
| DeepSeek-R1 (reasoning only) | — | — | — | canonicalize only |
| Kimi-K3 (XTML channels) | — | — | — | **out of scope**: per-call `index="N"` counter attribute, schema-derived `type=` attributes, attribute escaping, and response-channel wrapping exceed a declarative literal template; stays on its dedicated parser |
| gpt-oss (harmony) | — | — | — | stays on `openai_harmony` (already SSOT) |

Five args encodings cover eight recent families; each encoding's FSM
topology, grammar Format builder, and renderer are written once. Notably
the `minimax_ns_xml` structural tag is net-new capability — upstream has
NO constrained-decoding path for M3 tool calls (Rust parser only, no
structural tag).

## Evidence (tests, all passing)

* `TestGeneratedParser`: generated Qwen3 config is behaviorally identical
  to the hand-written `qwen3_config()` on reasoning + tool fixtures; a
  hermes-style JSON spec parses with zero per-model parser code.
* `TestStructuralTag`: generated grammars accept exactly the renderer's
  bytes for auto/required/forced; reject schema violations, unknown tool
  names, and bare text under `required`.
* `TestCanonicalization`: free reasoning + user JSON schema / choice /
  regex as one grammar; R1's start-in-prompt shape; thinking-off
  passthrough; EOS-in-reasoning masked.
* `TestReferenceTemplate`: generated Jinja fragment renders
  byte-identically to the wire renderer.
* `TestThreeWayConsistency`: renderer → parser → grammar full loop.

## Migration path (upstream-sized steps)

1. Land `format_spec` + specs for engine-based families; register the
   generated `ParserEngineConfig` through the existing adapters
   (no serving-layer changes).
2. Point `structural_tag_registry` at `tag_gen` for spec'd models; the
   `reasoning=` parameter that today is hardcoded `False` becomes
   `wrap_with_reasoning`.
3. Generalize the harmony `_adjust_output_format` call site
   (`Parser.adjust_request`) to use `canonicalize_structured_outputs`
   whenever the model has a spec — this is the "constrain the full
   output" end state, and where the spec-decode boundary machinery stops
   being load-bearing.
4. Add template round-trip validation against model repos' templates
   (warn, then enforce for vLLM-bundled templates).
5. Long term: upstream the spec schema itself (xgrammar already hosts
   builtin model tags; a shared declarative spec is the natural meeting
   point — see RFC #39848).

## Known limitations (from the adversarial review)

A three-lens review (parser parity, byte agreement, maintainer critique)
produced 28 confirmed findings; the mechanical ones are fixed on this
branch (JSON-aware argument converter with prefix-stable streaming
output, wrapper closers as their own token-id terminals, forced/
start-in-prompt thinking override shared by parser and grammar,
JSON-encoded non-string `qwen_xml` values, bare-object schema
normalization, `content=None` handling). What remains, by design or as
future work:

* **Canonical-bytes stance.** The generated FSM parses the *canonical*
  wire (what the renderer emits and the grammar enforces). Legacy
  parsers are more lenient on unconstrained output (compact JSON
  `{"name":"x"`, reordered keys). Under constrained decoding the tag's
  `begin` literal forces canonical spacing, so grammar-on requests are
  unaffected; grammar-off migration needs either regex terminals or a
  per-family eval sign-off.
* **Coverage.** `json`/`qwen_xml` do not express section-wrapped formats
  (deepseek_v3's `<｜tool▁calls▁begin｜>` groups, minimax's
  `<minimax:tool_call>` sections need `section_begin/end` fields) or
  bare-JSON formats with no distinctive closer (llama3 json needs a
  brace-balancing body rule). K3 XTML and harmony channels stay on their
  dedicated implementations.
* **Grammar/parser asymmetries.** `qwen_xml` string values containing
  `</function>` are grammar-legal but parser-lossy (renderer now raises
  on them); grammar-legal `</think>` inside post-boundary content is
  absorbed by the parser's duplicate-marker transition. Both need a
  policy decision (exclude in grammar vs pass through in parser).
* **Integration layer.** Registration is deliberately unbuilt: a cached
  `make_spec_engine(spec)` factory (reading `chat_template_kwargs`, and
  generalizing the Qwen3-style "unpaired tool-start counts as reasoning
  end" token-id override), lazy-registry shims, a model→spec resolver,
  and routing `_apply_structural_tag`/`adjust_request` through
  `tag_gen`/`canonicalize` (note: always-constraining `tool_choice=auto`
  reverses the current strict-only policy and needs evals).
* **Behavioral deltas to quantify.** EOS masking during reasoning and
  constraining previously-unconstrained outputs change sampling; model
  evals required before any serving-path flip (per AGENTS.md).
* Legacy structural tags (`structures`/`triggers`) cannot be
  reasoning-wrapped; only the format-based spec composes.
