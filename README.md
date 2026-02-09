# AI Council (ai-consil)

Multi-agent deliberation with **blind voting** via OpenAI-compatible API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

AI Council orchestrates multiple AI agents to deliberate on questions and reach conclusions through structured discussion and blind voting. Agents can have different roles (analyst, skeptic, advocate, etc.) and use different LLM providers.

**Key Features:**
- **Blind Voting**: Votes are sealed until voting closes - no agent can see others' votes during deliberation
- **OpenAI-Compatible API**: Drop-in replacement for `/v1/chat/completions`
- **Multi-Provider**: Use OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, or custom providers
- **Streaming**: Real-time council events via SSE
- **Configurable**: Rounds, question limits, voting schedules, consensus thresholds
- **Auditable**: Full transcript and vote ledger saved to `./out/`

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-consil.git
cd ai-consil

# Install with pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quickstart

### 1. Start the server

```bash
ai-consil serve
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 2. Make a request

#### Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-consil",
    "messages": [
      {"role": "user", "content": "Should we migrate our monolith to microservices?"}
    ],
    "stream": false,
    "council": {
      "agents": [
        {"id": "analyst", "role": "analyst", "provider": "mock", "model": "mock-v1"},
        {"id": "skeptic", "role": "skeptic", "provider": "mock", "model": "mock-v1"},
        {"id": "advocate", "role": "advocate", "provider": "mock", "model": "mock-v1"}
      ],
      "rounds": 2,
      "max_questions_per_agent": 2,
      "voting_schedule": "each_round",
      "trace": true
    }
  }'
```

#### Streaming request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "ai-consil",
    "messages": [
      {"role": "user", "content": "What is the best approach for rate limiting?"}
    ],
    "stream": true,
    "council": {
      "agents": [
        {"id": "security", "role": "domain_expert", "system_prompt": "You are a security engineer.", "provider": "mock", "model": "mock-v1"},
        {"id": "backend", "role": "pragmatist", "provider": "mock", "model": "mock-v1"}
      ],
      "rounds": 1,
      "voting_schedule": "end_only",
      "trace": true
    }
  }'
```

#### Using a config file

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-consil",
    "messages": [{"role": "user", "content": "Your question here"}],
    "council": "./configs/example_council.json"
  }'
```

### 3. CLI usage

```bash
# Run a deliberation from command line
ai-consil run "Should we use microservices?" --config configs/example_council.json

# With streaming output
ai-consil run "What database should we use?" -c configs/example_council.json --stream

# List available providers
ai-consil providers
```

## Configuration

### Council Config Schema

```json
{
  "agents": [
    {
      "id": "unique-agent-id",
      "role": "analyst|skeptic|advocate|pragmatist|innovator|domain_expert|synthesizer|ethicist",
      "system_prompt": "Optional custom prompt",
      "provider": "openai|anthropic|gemini|groq|deepseek|mock",
      "model": "gpt-4o|claude-sonnet-4-20250514|gemini-1.5-pro|..."
    }
  ],
  "rounds": 2,
  "max_questions_per_agent": 2,
  "voting_schedule": "each_round|start_end|end_only",
  "consensus_threshold": 0.67,
  "trace": true
}
```

### Built-in Roles

| Role | Description |
|------|-------------|
| `analyst` | Objective analysis, breaks down problems systematically |
| `skeptic` | Critical evaluation, identifies risks and flaws |
| `advocate` | Argues for feasibility and benefits |
| `pragmatist` | Focuses on practical implementation |
| `innovator` | Proposes creative alternatives |
| `domain_expert` | Deep technical expertise (use with custom `system_prompt`) |
| `synthesizer` | Identifies common ground, bridges viewpoints |
| `ethicist` | Considers moral and ethical implications |

### Provider Configuration

Set environment variables for each provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# Groq
export GROQ_API_KEY="gsk_..."

# DeepSeek
export DEEPSEEK_API_KEY="..."
```

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completions with council extensions.

**Request:**
```json
{
  "model": "ai-consil",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": null,
  "council": { ... }
}
```

**Response (non-streaming):**
```json
{
  "id": "council-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "ai-consil",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "council_trace": { ... }
}
```

**Response (streaming):**
```
data: {"id":"council-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"council-xxx","object":"council.event","event_type":"session_start",...}
data: {"id":"council-xxx","object":"council.event","event_type":"round_start","round":1,...}
data: {"id":"council-xxx","object":"council.event","event_type":"agent_analysis","agent_id":"analyst",...}
data: {"id":"council-xxx","object":"council.event","event_type":"voting_closed","round":1,...}
data: {"id":"council-xxx","object":"council.event","event_type":"vote_reveal","round":1,"votes":[...],...}
data: {"id":"council-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"The council..."}}]}
data: [DONE]
```

### GET /v1/providers

List available LLM providers.

### GET /v1/council/sessions

List all council sessions.

### GET /v1/council/sessions/{session_id}/transcript

Get the event transcript for a session.

### GET /v1/council/sessions/{session_id}/votes

Get the vote ledger for a session.

## Blind Voting Guarantee

AI Council implements strict blind voting:

1. **Vote Vault**: Votes are stored in a sealed vault that only releases votes after `close_voting()` is called
2. **Event Sanitizer**: All streaming events pass through a sanitizer that blocks vote data before voting closes
3. **Context Sanitizer**: Agent prompts never include other agents' votes
4. **No Vote Reveal Before Close**: The `vote_reveal` event can only be emitted after `voting_closed`

This ensures agents cannot be influenced by knowledge of how others have voted.

## Output Artifacts

Each session creates artifacts in `./out/{session_id}/`:

```
./out/council-abc123/
├── transcript.ndjson    # Event-sourced transcript (NDJSON)
├── vote_ledger.json     # Sealed votes and tallies
├── final_answer.md      # Synthesized answer
└── metadata.json        # Config and timestamps
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ai_consil

# Type checking
mypy src/ai_consil

# Linting
ruff check src/ai_consil
```

## Extending with Custom Providers

```python
from ai_consil.providers import register_provider, ProviderAdapter, ProviderMessage, ProviderResponse

class MyCustomProvider(ProviderAdapter):
    name = "my-provider"

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        # Your implementation
        return ProviderResponse(content="...")

    @classmethod
    def is_available(cls) -> bool:
        return True  # Check env vars, etc.

# Register the provider
register_provider("my-provider", MyCustomProvider, aliases=["mp"])

# Now use it in config
# {"provider": "my-provider", "model": "..."}
```

## Genesis

This project was created with the following prompt:

> You are an expert Python engineer. Help me build an open-source (MIT) project named **ai-consil** using the Strands Agents Python SDK.
>
> **Non-negotiables:**
> - **BLIND VOTING ONLY**: In any round where voting occurs, agents MUST NOT see any other agent's vote, vote tally, or partial results until AFTER voting is closed for that round. Orchestrator is NON-VOTING (zero voting power) and only coordinates + summarizes.
> - Agents can ask DIRECTED questions to specific agents, up to configured limits.
> - All Q&A is visible to all agents (but votes remain hidden until reveal).
> - Multi-round execution with configurable rules: number of rounds, max number of questions per agent (per round), voting schedule (each round OR start+end OR end-only).
> - Inputs: user prompt is text, plus optional references (images/URLs/videos) included as references in the input. For v0.1 there is NO web browsing/tooling; URLs/videos are just text references.
> - Output artifacts: final answer + vote results + ledger/transcript as JSON in ./out.
>
> **CRITICAL PRODUCT REQUIREMENT: API-FIRST + STANDARD CHAT COMPLETIONS + STREAMING**
> The project MUST expose an HTTP API so clients can use it like a normal Chat Completions endpoint and see the council "reasoning" (meaning the council transcript/events) via standard chat + streaming.
> - Implement an OpenAI-compatible endpoint: `POST /v1/chat/completions`
>   - Accepts: model, messages[], stream:boolean, temperature, max_tokens, and an extra field `council` (config inline OR pointer to config file)
>   - Returns: OpenAI-like response JSON for non-stream
>   - For stream=true: stream Server-Sent Events (SSE) with `data: {json}\n\n` chunks similar to OpenAI
> - The "reasoning" MUST be exposed as COUNCIL TRACE EVENTS, not hidden chain-of-thought.
>   - During streaming, emit intermediate council events (analysis summaries, Q&A, vote reveal) as structured chunks.
>   - Ensure BLIND VOTING: NEVER stream votes/tallies before the vote is closed and reveal is allowed.
> - Keep the final assistant message content as the final consolidated answer from orchestrator.
> - Also include an optional `trace` output:
>   - If request includes `council.trace=true`, include `council_trace` in response (non-stream) or emit trace chunks (stream).
>   - Trace must include event types + timestamps + agent ids + payloads, but must not leak votes early.
>
> **Tech stack (Python):** Python 3.11+, Strands Agents (Python SDK), Pydantic v2, FastAPI + Uvicorn, Typer for CLI, Pytest for tests.

## License

MIT License - see [LICENSE](LICENSE) for details.
