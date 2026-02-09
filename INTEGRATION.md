# AI Consil - Integration Guide

AI Consil is a multi-agent deliberation system with blind voting,, exposed as an OpenAI-compatible REST API. Multiple AI agents analyze a topic, ask each other questions, and vote — producing a synthesized answer.

## Quick Start

### 1. Install

```bash
git clone <repo-url> && cd ai-consil
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the server

```bash
ai-consil serve --port 8000
```

### 4. Make your first request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-consil",
    "messages": [{"role": "user", "content": "Should I invest in NVIDIA?"}],
    "council": {
      "agents": [
        {"id": "analyst", "role": "analyst", "provider": "openai", "model": "gpt-4o"},
        {"id": "skeptic", "role": "skeptic", "provider": "anthropic", "model": "claude-sonnet-4-20250514"}
      ],
      "rounds": 1,
      "vote_options": ["bullish", "neutral", "bearish"]
    }
  }'
```

---

## API Reference

### POST `/v1/chat/completions`

The main endpoint. OpenAI-compatible with a `council` extension field.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"ai-consil"` | Model identifier (informational) |
| `messages` | Message[] | *required* | Conversation messages. The last `user` message becomes the deliberation topic. |
| `stream` | boolean | `false` | Enable SSE streaming |
| `temperature` | float | `0.7` | Temperature (0.0-2.0). Ignored for OpenAI reasoning models (o1/o3/o4/o5). |
| `max_tokens` | integer | `null` | Max tokens per completion |
| `council` | object/string/null | `null` | Council config (inline object, path to JSON file, or null) |

**Message format:**

```json
{"role": "system|user|assistant", "content": "..."}
```

**Non-streaming response:**

```json
{
  "id": "council-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "ai-consil",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The council's synthesized answer..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "council_trace": { ... }
}
```

The `council_trace` field is included only when `trace: true` is set in the council config.

---

### GET `/health`

```json
{"status": "healthy"}
```

### GET `/v1/providers`

Lists registered providers and which ones have API keys configured.

```json
{
  "providers": ["mock", "openai", "anthropic", "gemini", "groq", "deepseek"],
  "available": ["mock", "openai", "anthropic"]
}
```

### GET `/v1/council/sessions`

```json
{"sessions": ["council-abc123", "council-def456"]}
```

### GET `/v1/council/sessions/{session_id}/transcript`

```json
{
  "session_id": "council-abc123",
  "events": [ ... ]
}
```

### GET `/v1/council/sessions/{session_id}/votes`

Returns the vote ledger (rounds, individual votes, tallies, consensus status).

---

## Council Configuration

The `council` field in the request body configures the deliberation. It can be:
- An **inline JSON object**
- A **file path** to a JSON config file (e.g., `"configs/financial_council.json"`)
- `null` to skip council deliberation

### Full schema

```json
{
  "agents": [
    {
      "id": "unique-agent-id",
      "role": "analyst",
      "system_prompt": "Custom system prompt for this agent",
      "provider": "openai",
      "model": "gpt-4o",
      "provider_config": {
        "web_search": true
      }
    }
  ],
  "rounds": 2,
  "max_questions_per_agent": 2,
  "voting_schedule": "each_round",
  "consensus_threshold": 0.67,
  "vote_options": ["support", "oppose", "abstain"],
  "trace": false
}
```

### Agent fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | *required* | Unique identifier for the agent |
| `role` | string | *required* | Role type: `analyst`, `skeptic`, `advocate`, `pragmatist`, `innovator`, `domain_expert`, `synthesizer`, `ethicist` |
| `system_prompt` | string | `null` | Custom system prompt. If not set, a default prompt is generated from the role. |
| `provider` | string | `"openai"` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `deepseek`, `mock` |
| `model` | string | `"gpt-4o"` | Model identifier (e.g., `gpt-4o`, `o3`, `claude-sonnet-4-20250514`) |
| `provider_config` | object | `{}` | Provider-specific settings (see below) |

### Provider config options

| Option | Type | Providers | Description |
|--------|------|-----------|-------------|
| `web_search` | boolean | OpenAI, Anthropic | Enable web search for the agent |
| `max_searches` | integer | Anthropic | Max web searches per completion (default: 5) |

### Council settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rounds` | integer | `2` | Number of deliberation rounds (1-10) |
| `max_questions_per_agent` | integer | `2` | Questions each agent can ask per round (0-10, 0 disables Q&A) |
| `voting_schedule` | string | `"each_round"` | When voting occurs: `each_round`, `start_end`, `end_only` |
| `consensus_threshold` | float | `0.67` | Fraction of votes needed for consensus (0.0-1.0) |
| `vote_options` | string[] | `["support", "oppose", "abstain"]` | Custom vote options |
| `trace` | boolean | `false` | Include full deliberation trace in response |

---

## Streaming (SSE)

Set `"stream": true` to receive Server-Sent Events. Events follow this lifecycle:

```
session_start -> round_start -> agent_analysis (per agent)
  -> question/answer (Q&A phase)
  -> voting_open -> voting_closed -> vote_reveal
  -> round_end -> [next round...] -> synthesis -> session_end
```

### Event format

Each SSE line is `data: {json}\n\n`. The JSON has this structure:

```json
{
  "id": "council-abc123",
  "object": "council.event",
  "event_type": "agent_analysis",
  "timestamp": "2025-01-15T10:30:00Z",
  "round": 1,
  "agent_id": "analyst-1",
  "from_agent": null,
  "to_agent": null,
  "content": "Based on my analysis...",
  "votes": null,
  "tally": null,
  "error": null,
  "metadata": null
}
```

### Event types

| Event | Key Fields | Description |
|-------|-----------|-------------|
| `session_start` | `content` (topic), `metadata.agent_count` | Deliberation begins |
| `round_start` | `round`, `metadata.agents` | New round starts |
| `agent_analysis` | `round`, `agent_id`, `content` | Agent provides analysis |
| `question` | `round`, `from_agent`, `to_agent`, `content` | Agent asks a question |
| `answer` | `round`, `from_agent`, `to_agent`, `content` | Agent answers a question |
| `voting_open` | `round` | Voting opens (blind) |
| `voting_closed` | `round` | Voting closes, votes are sealed |
| `vote_reveal` | `round`, `votes[]`, `tally`, `metadata.consensus` | Votes revealed after close |
| `round_end` | `round` | Round complete |
| `synthesis` | `content` | Final synthesized answer |
| `session_end` | `content`, `metadata.consensus` | Deliberation complete |
| `error` | `error` | Error occurred |

### Vote reveal payload

```json
{
  "votes": [
    {"agent_id": "analyst", "position": "bullish", "confidence": 0.85, "reasoning": "Strong momentum..."},
    {"agent_id": "skeptic", "position": "neutral", "confidence": 0.7, "reasoning": "Valuation concerns..."}
  ],
  "tally": {
    "counts": {"bullish": 1, "neutral": 1, "bearish": 0}
  }
}
```

### Blind voting guarantee

Vote data (`votes`, `tally`) is **never** included in events before `voting_closed`. The `BlindVotingSanitizer` strips any vote information from streaming events to prevent agents from being influenced by each other's votes.

---

## Custom Vote Options

By default, agents vote `support`, `oppose`, or `abstain`. You can customize this per session:

```json
{
  "vote_options": ["bullish", "neutral", "bearish"]
}
```

Or for a different use case:

```json
{
  "vote_options": ["approve", "revise", "reject"]
}
```

Agents will be prompted with these options and their votes are validated against the list. Consensus is calculated using plurality: the top option's share of total votes vs. the `consensus_threshold`.

---

## Provider Setup

Each provider requires its API key as an environment variable:

| Provider | Env Variable | Models (examples) |
|----------|-------------|-------------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| Gemini | `GOOGLE_API_KEY` | `gemini-2.0-flash`, `gemini-2.5-pro` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |
| Mock | *(none)* | `mock-v1` (returns canned responses, for testing) |

Provider aliases: `"gpt"` = openai, `"claude"` = anthropic, `"google"` = gemini, `"ds"` = deepseek.

---

## Code Examples

### Python (requests)

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "ai-consil",
    "messages": [{"role": "user", "content": "Should I invest in NVIDIA?"}],
    "council": {
        "agents": [
            {"id": "analyst", "role": "analyst", "provider": "openai", "model": "gpt-4o"},
            {"id": "skeptic", "role": "skeptic", "provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        ],
        "rounds": 1,
        "vote_options": ["bullish", "neutral", "bearish"],
        "trace": True
    }
})

data = response.json()
print(data["choices"][0]["message"]["content"])
```

### Python (httpx streaming)

```python
import httpx
import json

with httpx.stream("POST", "http://localhost:8000/v1/chat/completions", json={
    "model": "ai-consil",
    "messages": [{"role": "user", "content": "Should I invest in NVIDIA?"}],
    "stream": True,
    "council": {
        "agents": [
            {"id": "analyst", "role": "analyst", "provider": "openai", "model": "gpt-4o"},
            {"id": "skeptic", "role": "skeptic", "provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        ],
        "rounds": 1,
        "vote_options": ["bullish", "neutral", "bearish"],
        "trace": True
    }
}) as response:
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            event = json.loads(line[6:])
            if event.get("object") == "council.event":
                print(f"[{event['event_type']}] {event.get('content', '')[:100]}")
```

### curl (streaming)

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-consil",
    "stream": true,
    "messages": [{"role": "user", "content": "Should I invest in NVIDIA?"}],
    "council": {
      "agents": [
        {"id": "analyst", "role": "analyst", "provider": "openai", "model": "gpt-4o"},
        {"id": "skeptic", "role": "skeptic", "provider": "anthropic", "model": "claude-sonnet-4-20250514"}
      ],
      "rounds": 1,
      "vote_options": ["bullish", "neutral", "bearish"]
    }
  }'
```

### Using a config file

```bash
# Non-streaming with config file
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-consil",
    "messages": [{"role": "user", "content": "Should I invest in NVIDIA?"}],
    "council": "configs/financial_council.json"
  }'
```

---

## CLI Usage

```bash
# Start API server
ai-consil serve --host 0.0.0.0 --port 8000 --reload

# Run deliberation directly from CLI
ai-consil run "Should I invest in NVIDIA?" -c configs/financial_council.json --stream

# List available providers
ai-consil providers

# Show version
ai-consil version
```

---

## Example Configs

See `configs/` directory for ready-to-use configurations:

- `configs/financial_council.json` — 4 agents (2x OpenAI o3, 2x Claude Sonnet) with web search, bullish/neutral/bearish voting
- `configs/openai_anthropic_council.json` — 3 agents (2x GPT-4o, 1x Claude Sonnet), default voting

### Minimal config

```json
{
  "agents": [
    {"id": "a1", "role": "analyst", "provider": "mock", "model": "mock-v1"},
    {"id": "a2", "role": "skeptic", "provider": "mock", "model": "mock-v1"}
  ],
  "rounds": 1
}
```
