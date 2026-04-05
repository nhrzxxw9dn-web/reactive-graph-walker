# RGW — Reactive Graph Walker

A self-propagating cognitive engine. The graph drives its own computation through cascading edge activation. No loops. No timers. No ticks. The topology IS the clock.

## What is this?

RGW is an alternative to transformer-based AI. Instead of pattern matching against training data, it thinks through parallel graph traversal with emotional biasing. Multiple walkers explore a memory graph simultaneously — convergence means confidence, divergence means novelty.

The core primitive:

```
Signal + SelfModel → (Signal, SelfModel', Noticing)
```

One function. Everything else emerges from it. Goals form when the self-model notices patterns in its own behavior. Emotions modulate which edges walkers follow. The walk changes the graph. The graph IS the mind.

## Architecture

```
┌────────────────────────────────────────────┐
│  RGW (one process, one entity)             │
│                                            │
│  core.rs      — the primitive              │
│  diverger.rs  — self-propagating reactor   │
│  walker.rs    — parallel graph traversal   │
│  graph.rs     — emotional biasing          │
│  provider.rs  — multi-LLM routing          │
│  tools.rs     — web search, code exec      │
│  speech.rs    — TTS/STT                    │
│  motor.rs     — commands Julian's body     │
│  openai.rs    — /v1/chat/completions       │
│  db.rs        — PostgreSQL + pgvector      │
│  api.rs       — 14 HTTP endpoints          │
└────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build
cargo build --release

# Run (needs PostgreSQL with memory_vectors + memory_edges tables)
./target/release/rgw \
  --db-url postgresql://user:pass@localhost/julian \
  --ollama-url http://localhost:11434 \
  --julian-url http://localhost:8000 \
  --port 11435
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible (think + express) |
| `/v1/models` | GET | List available models |
| `/walk` | POST | Raw graph traversal |
| `/self` | GET | Self-model state (consciousness) |
| `/self/save` | POST | Persist self-model to DB |
| `/diverger` | GET | Reactive engine stats |
| `/diverger/notify` | POST | Edge change notification |
| `/edge` | POST | Create new graph edge |
| `/prune` | POST | Synaptic pruning (decay + delete) |
| `/tools` | GET | List available tools |
| `/tools/execute` | POST | Execute a tool |
| `/speak` | POST | Text-to-speech |
| `/stats` | GET | Graph topology |
| `/benchmark` | GET | Performance test |
| `/health` | GET | Status check |

## How It Works

1. **Memory changes in Julian's Python backend** notify the Diverger via `/diverger/notify`
2. **Energy propagates** to neighboring nodes through edges
3. **When a node's energy crosses its threshold**, a spontaneous walk fires
4. **Parallel walkers** traverse the graph with different emotional biases (fear, curiosity, experience, random)
5. **Each step passes through the self-model** — the walk is self-aware
6. **If the walk produces an actionable result**, a motor command is sent to Julian
7. **Julian acts** (tweets, blogs, journals) and stores new memories
8. **New memories create new edge changes** — the cycle continues

No external scheduler. The graph sustains its own cognitive activity through cascading activation.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | required | PostgreSQL connection string |
| `WALKER_PORT` | 11435 | HTTP server port |
| `WALKER_HOST` | 127.0.0.1 | HTTP server bind address |
| `WALKER_THREADS` | auto | Rayon thread count (0 = all cores) |
| `OLLAMA_URL` | http://localhost:11434 | Qwen for text expression |
| `EXPRESSION_MODEL` | qwen3:14b | Model name for expression |
| `JULIAN_URL` | http://localhost:8000 | Julian's Python backend |

## Performance

Target: 1000+ walks/second on Mac M1 Pro (10 cores).
Binary size: ~3MB.
Memory: proportional to graph size (~20MB for 10K nodes).

Hit `/benchmark` to measure your hardware.

## License

Part of Project Julian.
