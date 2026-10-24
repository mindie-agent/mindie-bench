# Strategy selector corpus fixtures

These are verbatim committed public C5 units for the four selector candidates
and their inheritance closure. All unit `.md` files under this directory are
frozen evaluation inputs (retaining their original relative paths, frontmatter,
and unit provenance) and must not be reworded or edited as documentation prose.

Scope and evidence boundary:
- These fixtures exercise isolated Arm B / Arm C routing and accounting contracts
  in `mindie_bench.evaluation.strategy_selection`; they do not establish
  full-corpus retrieval quality or model/NPU benefits.
- The shipped default runs only the selector stage (`continuation="not_executed"`).
  Historical quantitative continuation conclusions are suspended from serving as
  current acceptance evidence, and unmeasured continuation outcomes or resource
  costs (`total_wall_clock_sec`, `total_model_calls`, `npu_card_hours`,
  `npu_die_hours`) remain `None` rather than being filled by synthetic
  continuations or planned-device defaults.

Sources:
- `knowledge-vllm-ascend` at `d3d5bf6320e171d4a0f131a6f746858700e0c4cc`: https://github.com/mindie-agent/knowledge-vllm-ascend/tree/d3d5bf6320e171d4a0f131a6f746858700e0c4cc
- `knowledge-operators` at `6f083f810c0d8eb77b9d7992ba96b19c51dfaf74`: https://github.com/mindie-agent/knowledge-operators/tree/6f083f810c0d8eb77b9d7992ba96b19c51dfaf74
