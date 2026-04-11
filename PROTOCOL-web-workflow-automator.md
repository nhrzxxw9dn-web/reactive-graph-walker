# Protocol: RGW as Web Workflow Automator

**Date**: 2026-04-11
**Scope**: Adapting RGW from persona simulator to autonomous web presence manager
**Status**: Architecture implemented, backend integration pending
**Depends on**: PROTOCOL-compliance-mode.md (CognitiveMode)

---

## Executive Summary

RGW's architecture — decoupled brain/body, reactive graph traversal, friction-based learning,
metacognitive safety — maps 1:1 to the requirements for automating a small web project
(blog, images, social media). This document specifies how each RGW subsystem serves each
requirement, what was built, and what remains.

---

## Architecture Mapping

### Blueprint Requirement → RGW Implementation

| Blueprint | RGW Component | Status |
|-----------|---------------|--------|
| Decoupled brain/body | `diverger.rs` (brain) + `motor.rs` (body) | Built |
| Standardized motor commands | `MotorCommand` with flexible `params: Option<Value>` | Built (this session) |
| Commoditized LLM routing | `provider.rs` (local/cloud routing by complexity) | Built, now wired in |
| Multi-modal tooling | `tools.rs` (image_generate, post_social, blog_publish) | Built (this session) |
| Agentic friction on failure | `friction.rs` (errors → pain signals → learning) | Built |
| Metacognitive safety pause | `metacog.rs` (critique before action) | Built |
| Compliance mode for strict tasks | `core.rs` CognitiveMode::Compliant | Built (this session) |
| External signal ingest | `POST /ingest/signal` endpoint | Built (this session) |

---

## What Was Built (This Session)

### 1. Expanded Motor Cortex (`src/motor.rs`)

**Change**: Added `params: Option<serde_json::Value>` to `MotorCommand`.

This enables the Rust brain to dictate action-specific structured data that the Python
body interprets. The brain decides "generate an image of a sunset in watercolor style."
The body translates that into the correct ComfyUI / DALL-E / Midjourney API call.

**Supported param schemas by action**:

```
image_generate:
  prompt: string      — Image generation prompt
  style: string       — "photorealistic", "illustration", "abstract"
  size: string        — "1024x1024", "512x512"
  provider: string    — "comfyui", "dalle", "midjourney"

post_social:
  platform: string    — "twitter", "bluesky", "mastodon", "linkedin"
  text: string        — Post content
  image_url: string   — Optional attached image
  schedule_at: string — ISO 8601 for scheduled posting

blog:
  title: string       — Blog post title
  body_md: string     — Post body in Markdown
  tags: [string]      — Categories/tags
  publish: bool       — Immediate publish or draft

email:
  to: string          — Recipient
  subject: string     — Subject line
  body: string        — Email body
```

### 2. New Tools (`src/tools.rs`)

**Added**: `image_generate`, `post_social`, `blog_publish`

All three are **motor-delegated tools** — they don't call external APIs directly.
They construct a `MotorCommand` and POST it to the execution backend. This preserves
the brain/body separation.

**Friction integration**: If the motor cortex is unreachable or rejects the command,
the tool returns a failure `ToolResult`, which `friction.rs` converts into:
- A pain signal through `core::process()`
- A capability wound (accumulates with repeated failures)
- An open question ("Why did image_generate fail?")

**Configuration**: Motor URL comes from `params.motor_url` or `RGW_MOTOR_URL` env var.

### 3. Provider Router Wired In (`src/openai.rs`)

**Change**: `chat_completions()` now routes through `provider.rs` before falling back to Ollama.

**Complexity derivation from walker output**:
```rust
complexity = (novelty_score * 0.6 + (1.0 - agreement_score) * 0.4).clamp(0.0, 1.0)
```

| Walk Result | Complexity | Route |
|------------|-----------|-------|
| High agreement, low novelty | ~0.2 | Local/cheap model (tweet draft, categorization) |
| Mixed | ~0.5 | Mid-tier (social post from graph context) |
| Low agreement, high novelty | ~0.8 | Premium cloud (blog synthesis from scattered memories) |

**Fallback chain**: Provider → Ollama → walker context as text.

### 4. Signal Ingest (`POST /ingest/signal`)

**New endpoint** for feeding external events into the cognitive loop.

```bash
# RSS feed item
curl -X POST http://localhost:3003/ingest/signal \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "rss",
    "content": "New article: AI agents are replacing social media managers",
    "domain": "ai_news",
    "intensity": 0.7
  }'

# Inbound email
curl -X POST http://localhost:3003/ingest/signal \
  -d '{"kind": "email", "content": "Client wants blog post about Q2 results", "domain": "client_work", "intensity": 0.8}'

# Social mention
curl -X POST http://localhost:3003/ingest/signal \
  -d '{"kind": "social_mention", "content": "@julian great thread on graph databases", "domain": "social", "intensity": 0.4}'

# Analytics alert
curl -X POST http://localhost:3003/ingest/signal \
  -d '{"kind": "analytics", "content": "Blog traffic up 300% on AI post", "domain": "content_performance", "intensity": 0.6}'
```

**Response** includes the self-model's state after processing — what the agent noticed,
whether focus shifted, current emotional state.

### 5. AppState Extended

`provider: Option<Provider>` added to shared application state. Initialized at startup
with `ProviderConfig::default()`. If no cloud models configured, falls back to Ollama
transparently.

---

## Complete Workflow: Blog + Image + Social

Here's how the full pipeline works end-to-end:

### Phase 1: Signal Arrives
```
External RSS poller → POST /ingest/signal
  kind: "rss"
  content: "OpenAI releases GPT-5 with native agent capabilities"
  domain: "ai_news"
  intensity: 0.7
```

The signal flows through `core::process()`. In Autonomous mode:
- Self-model arousal increases (novelty signal)
- Attention pattern for "ai_news" strengthens
- If this domain was already active, diverger may fire a spontaneous walk

### Phase 2: Graph Walk Produces Insight
The walk traverses the knowledge graph, finding connections between "ai_news",
"graph_databases", "agent_architecture" (existing nodes). Agreement is low (novel topic),
novelty is high (cross-domain connections).

### Phase 3: Metacognitive Critique
`metacog::critique()` evaluates:
- Safety: energy > 0.15 ✓
- Hallucination: agreement > threshold ✓
- Efficiency: not repeating recent topic ✓
- Result: **approved**

### Phase 4: Motor Command → Blog Draft
```
MotorCommand {
  action: "blog",
  domain: "ai_news",
  walker_context: "... cross-domain insight about agent architectures ...",
  params: {
    "title": "Why Graph-Driven Agents Will Replace Chain-of-Thought",
    "tags": ["ai", "agents", "architecture"],
    "publish": false  // Draft first
  }
}
```

LLM routing: complexity = 0.82 → "creative" strength → routes to Claude/Gemini.

### Phase 5: Image Generation
```
MotorCommand {
  action: "image_generate",
  params: {
    "prompt": "Abstract neural network graph with glowing nodes, dark background, digital art",
    "style": "illustration",
    "provider": "comfyui"
  }
}
```

### Phase 6: Social Distribution
```
MotorCommand {
  action: "post_social",
  params: {
    "platform": "twitter",
    "text": "New post: Why Graph-Driven Agents Will Replace CoT...",
    "image_url": "/generated/blog-hero.png"
  }
}
```

LLM routing: complexity = 0.3 → "fast" strength → routes to local Qwen or Groq.

### Phase 7: Friction / Success
- If image gen times out → `friction::tool_friction()` → wound on "image_generate"
- If tweet posts successfully → `friction::affordance_confirmed("post_social")`
- If blog fails to publish → `friction::motor_friction("blog", error)` → arousal spike

---

## Deployment Configuration

### Minimum (Ollama-only, no cloud)
```bash
RGW_MOTOR_URL=http://localhost:5000  # Python backend
cargo run -- --ollama-url http://localhost:11434 --julian-url http://localhost:5000
```

### Production (multi-provider routing)
Provider config loaded at startup. Example `provider.toml`:
```toml
cloud_threshold = 0.5  # Below 0.5 = local, above = cloud

[[cloud_models]]
provider = "groq"
model = "llama-3.3-70b-versatile"
api_key = "$GROQ_API_KEY"
strength = "fast"
cost_per_million = 0.59
priority = 1

[[cloud_models]]
provider = "anthropic"
model = "claude-sonnet-4-6"
api_key = "$ANTHROPIC_API_KEY"
strength = "creative"
cost_per_million = 3.0
priority = 2

[[cloud_models]]
provider = "gemini"
model = "gemini-2.5-pro"
api_key = "$GEMINI_API_KEY"
strength = "reasoning"
cost_per_million = 1.25
priority = 3
```

### Cognitive Mode for Different Workflows
```bash
# Autonomous: brainstorming, content discovery, spontaneous posting
curl -X POST http://localhost:3003/self/mode -d '{"mode": "autonomous"}'

# Compliant: scheduled content calendar, client deliverables
curl -X POST http://localhost:3003/self/mode -d '{"mode": "compliant"}'

# Per-request: compliant blog from an autonomous system
curl -X POST http://localhost:3003/v1/chat/completions \
  -d '{"messages": [...], "rgw_mode": "compliant"}'
```

---

## What Remains (Backend / Integration)

### Python Motor Backend
The Rust brain is complete. The Python body needs handlers for each action:

| Action | Python Handler | External API |
|--------|---------------|-------------|
| `tweet` / `post_social` | `handle_post_social(params)` | Twitter API v2, Bluesky AT Protocol |
| `blog` / `blog_publish` | `handle_blog_publish(params)` | WordPress REST API, GitHub Pages, Hugo |
| `image_generate` | `handle_image_generate(params)` | ComfyUI local API, DALL-E API, Replicate |
| `search` | `handle_search(params)` | Tavily, SerpAPI, or existing web_search tool |
| `email` | `handle_email(params)` | SMTP / SendGrid / AWS SES |

Each handler should:
1. Execute the API call
2. Return success/failure response (so `ToolResult` flows back through friction)
3. On success, optionally POST back to `/ingest/signal` (e.g., "blog published successfully")

### Signal Polling Service
A lightweight service (Python or shell cron) that polls external sources and POSTs
to `/ingest/signal`:

```python
# RSS poller (runs every 15 minutes)
for entry in feedparser.parse("https://news.example.com/feed").entries:
    requests.post("http://localhost:3003/ingest/signal", json={
        "kind": "rss",
        "content": f"{entry.title}: {entry.summary[:200]}",
        "domain": "industry_news",
        "intensity": 0.5,
    })
```

### Provider Config Loading
Currently `ProviderConfig::default()` creates an empty config (no cloud models).
Need to load from a config file (`provider.toml` or env vars) at startup.

### Database Schema for CognitiveMode
`save_self_model()` and `restore_self_model()` need to serialize/deserialize
the `mode` field. Currently not persisted — mode resets to Autonomous on restart.

---

## Cost Model

Running 24/7 with this architecture:

| Component | Cost | Notes |
|-----------|------|-------|
| Rust engine (diverger, walker, core) | ~$5/mo VPS | CPU-bound, no GPU needed |
| Local Qwen (simple tasks) | $0 | Runs on same VPS via Ollama |
| Groq (fast tasks, ~100/day) | ~$2/mo | 3000 tokens avg × 100 requests |
| Claude/Gemini (complex, ~10/day) | ~$3/mo | 5000 tokens avg × 10 requests |
| ComfyUI (local image gen) | $0 | If GPU available, else DALL-E ~$0.04/image |
| Total | **~$10-15/mo** | Full autonomous web presence |

Compare: running GPT-4 for every request at 24/7 = $500+/mo.

---

## Files Modified (This Session — Web Workflow)

| File | Change |
|------|--------|
| `src/motor.rs` | Added `params: Option<Value>`, expanded action docs |
| `src/tools.rs` | Added `image_generate`, `post_social`, `blog_publish` tools + `motor_tool()` |
| `src/openai.rs` | Wired provider router, complexity-based LLM routing |
| `src/api.rs` | Added `provider` to AppState, `POST /ingest/signal`, provider init |
| `src/diverger.rs` | Added `params: None` to MotorCommand construction |

## Files Modified (Previous Session — Compliance Mode)

| File | Change |
|------|--------|
| `src/core.rs` | CognitiveMode enum, mode field in SelfModel, gated processing |
| `src/metacog.rs` | Mode-aware critique (higher bar in compliant, no wound/efficiency gating) |
| `src/diverger.rs` | Suppress spontaneous walks in compliant mode |
| `src/dream.rs` | Disable dreaming in compliant mode |
| `src/walker.rs` | Restrict biases, use compliant edge scoring |
| `src/graph.rs` | Added `score_edge_compliant()` |
| `src/api.rs` | Per-request mode override, `POST /self/mode` |
| `src/openai.rs` | Per-request mode override in chat completions |
