# AI Consil - Project Guide

Multi-agent deliberation system with blind voting, exposed via an OpenAI-compatible API.

## Tech Stack

- Python 3.11+, FastAPI, Pydantic v2, asyncio, Typer CLI
- Providers: OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mock
- Build: hatchling, pytest, ruff, mypy

## Project Structure

```
src/ai_consil/
  main.py              # FastAPI app + Typer CLI entry point
  api/
    routes.py           # 6 HTTP endpoints (OpenAI-compatible)
    schemas.py          # All Pydantic models (request/response/events)
  council/
    orchestrator.py     # Main deliberation loop (rounds, Q&A, voting, synthesis)
    agent.py            # CouncilAgent wrapper (analyze, vote, ask/answer)
    vote_vault.py       # Sealed vote storage (blind voting enforcement)
    sanitizer.py        # BlindVotingSanitizer (strips vote data before reveal)
    roles.py            # System prompts and prompt templates
  providers/
    base.py             # ProviderAdapter abstract base + ProviderRegistry
    registry.py         # Global registry, get_provider(), register_provider()
    openai_provider.py  # OpenAI adapter (Chat Completions + Responses API for web search)
    anthropic_provider.py # Anthropic adapter (Messages API + web_search tool)
    gemini_provider.py  # Google Gemini adapter
    groq_provider.py    # Groq adapter
    deepseek_provider.py # DeepSeek adapter
    mock.py             # Mock provider for testing
  config/
    loader.py           # load_config_from_file(), resolve_config(), get_default_config()
  storage/
    artifacts.py        # ArtifactStore for session transcripts, vote ledgers
  streaming/
    sse.py              # SSE event formatting for streaming responses
configs/                # Example council JSON configs
tests/                  # 69 tests (pytest, asyncio_mode=auto)
```

## Key Patterns

- **Provider adapter pattern**: All LLM providers implement `ProviderAdapter` base class with `complete()` and `stream()` methods. Registered via `ProviderRegistry` with aliases (e.g., "openai" or "gpt").
- **Blind voting**: `VoteVault` stores sealed votes; `BlindVotingSanitizer` strips vote data from SSE events before voting is closed. Vote data is only revealed after `close_voting()`.
- **Dynamic vote options**: `vote_options` in config (e.g., `["bullish", "neutral", "bearish"]`). Defaults to `["support", "oppose", "abstain"]`.
- **Consensus**: Plurality-based. Top vote count / total votes >= `consensus_threshold`.
- **Web search**: Enabled per-agent via `provider_config: {"web_search": true}`. OpenAI uses Responses API with `web_search_preview` tool; Anthropic uses `web_search_20250305` tool.
- **Reasoning models**: OpenAI o-series models (o1, o3, o4, o5) skip temperature parameter automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Main deliberation endpoint (streaming + non-streaming) |
| GET | `/health` | Health check |
| GET | `/v1/providers` | List available providers |
| GET | `/v1/council/sessions` | List session IDs |
| GET | `/v1/council/sessions/{id}/transcript` | Session event transcript |
| GET | `/v1/council/sessions/{id}/votes` | Session vote ledger |

## Council Config Schema

```json
{
  "agents": [
    {
      "id": "agent-1",
      "role": "analyst|skeptic|advocate|pragmatist|innovator|domain_expert|synthesizer|ethicist",
      "system_prompt": "Custom prompt (optional)",
      "provider": "openai|anthropic|gemini|groq|deepseek|mock",
      "model": "gpt-4o|claude-sonnet-4-20250514|o3|...",
      "provider_config": {"web_search": true}
    }
  ],
  "rounds": 2,
  "max_questions_per_agent": 2,
  "voting_schedule": "each_round|start_end|end_only",
  "consensus_threshold": 0.67,
  "vote_options": ["support", "oppose", "abstain"],
  "trace": false
}
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Start API server
ai-consil serve --host 0.0.0.0 --port 8000

# Run deliberation from CLI
ai-consil run "Your question here" -c configs/financial_council.json --stream

# List providers
ai-consil providers
```

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI provider
- `ANTHROPIC_API_KEY` - Required for Anthropic provider
- `GOOGLE_API_KEY` - Required for Gemini provider
- `GROQ_API_KEY` - Required for Groq provider
- `DEEPSEEK_API_KEY` - Required for DeepSeek provider

## Testing

```bash
pytest                      # All 69 tests
pytest tests/test_blind_voting.py  # Blind voting tests (critical)
pytest tests/test_api.py    # API endpoint tests
pytest -x -v               # Stop on first failure, verbose
```
