# REPORT

## Phase 1 - vLLM serving configuration

### Endpoint and model
- Endpoint: `http://localhost:8000`
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Hardware: `1x H100 80GB`

Flags and one-line justification:
- `--model Qwen/Qwen3-30B-A3B-Instruct-2507`: fixed assignment model.
- `--host 0.0.0.0`: expose server for local tooling and dashboards.
- `--port 8000`: required endpoint for this assignment flow.
- `--max-model-len 8096`: enough context headroom for 1.5-3K token prompts plus schema and completion.
- `--tensor-parallel-size 1`: single H100, no tensor sharding needed.
- `--enable-expert-parallel`: helps MoE expert routing efficiency on this model family.
- `--enable-auto-tool-choice`: enables automatic tool-choice behavior support in OpenAI-compatible calls.
- `--tool-call-parser qwen3_coder`: parser tuned to Qwen3-style tool outputs.
- `--reasoning-parser qwen3`: parser for Qwen3 reasoning format.