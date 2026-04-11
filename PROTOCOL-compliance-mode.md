# Protocol: CognitiveMode (Compliance Mode)

**Date**: 2026-04-11
**Scope**: System-wide behavioral gating via `CognitiveMode` enum
**Status**: Implemented, compiles, untested at runtime

---

## Problem

RGW has five sources of non-determinism that make it unsuitable for compliance-critical deployment:

1. **Emotional signal coloring** — arousal, wounds, and competencies modify signal intensity
2. **Spontaneous diverger walks** — edge energy accumulation triggers unplanned graph traversals
3. **Dream edge creation** — Monte Carlo exploration permanently mutates graph topology
4. **Wound-biased metacognition** — past pain blocks correct actions in affected domains
5. **Belief formation** — accumulated noticings crystallize into opinions that influence processing

These are features for autonomous personas (Julian). They are liabilities for enterprise, customer service, or research assistant use cases.

## Solution

A `CognitiveMode` enum embedded in the `SelfModel` itself. The system is self-aware of its operating mode. The self-model still participates in every computation (structural requirement), but the mode determines whether internal state influences output.

```rust
pub enum CognitiveMode {
    Autonomous,  // Full emotional agency
    Compliant,   // Deterministic, task-focused
}
```

## Changes by file

### `src/core.rs` — The Primitive

**Added**: `CognitiveMode` enum (Autonomous, Compliant) with doc comments.

**Added**: `mode: CognitiveMode` field to `SelfModel`, initialized to `Autonomous`.

**Modified**: `process()` function — three gates:

1. **Signal influence (section 2)**: In Compliant mode, signals pass through at raw intensity. No arousal multiplier, no wound amplification, no competency boost. The self-model observes but does not color perception.

2. **Emotional state changes (section 3)**: In Compliant mode, no valence/arousal/energy drift. No wound accumulation. No attention pattern decay. The internal state is frozen — the professional at work.

3. **Pattern/belief formation (section 4)**: In Compliant mode, noticings still fire (the system still notices), but they don't accumulate into `noticings` vec, don't trigger `detect_patterns()`, and don't form beliefs. No opinion formation.

### `src/metacog.rs` — The Critic

**Modified**: `critique()` function:

- **Both modes**: Safety check (energy > 0.15) and hallucination check (walker agreement threshold) still apply.
- **Compliant mode**: Hallucination threshold raised from 0.2 to 0.3 (higher bar for factual confidence).
- **Autonomous only**: Efficiency check (domain saturation < 70%) and wound check (past pain + low confidence blocks action).
- **Compliant skips**: No domain saturation gating (answer the question even if we just answered one like it). No wound avoidance (do the task even if it hurts).

### `src/diverger.rs` — The Reactive Engine

**Modified**: `start_reactor()` inner loop:

- After computing `nodes_to_fire`, checks `self_model.mode`.
- **Compliant mode**: `continue` — energy still propagates and decays (the graph stays "alive"), but no spontaneous walks fire. The system only acts when explicitly asked via API.
- **Autonomous mode**: Unchanged behavior — threshold crossing triggers spontaneous walks.

### `src/dream.rs` — Monte Carlo Exploration

**Modified**: `dream()` function:

- **Compliant mode**: Early return with empty `DreamReport`. Graph topology is frozen. No Monte Carlo perturbation, no new edge creation.
- Logged: `[dream] Compliant mode — dreaming disabled, graph frozen`

### `src/walker.rs` — Graph Traversal

**Modified**: `walk_parallel()` — bias assignment:

- **Autonomous**: All biases (Fear, Curiosity, Experience, Random) — full perspective diversity.
- **Compliant**: Only `Experience` and `Analytical` biases. These follow strong, causal, and reinforcing edges. No Fear (no threat-seeking), no Curiosity (no weak-edge exploration), no Random (no noise), no Contrarian (no contradiction-seeking).

**Modified**: `walk_single()` — edge scoring:

- **Autonomous**: `score_edge()` with full emotional modulation (arousal multiplier, valence alignment, emotional charge influence).
- **Compliant**: `score_edge_compliant()` — pure weight-based scoring with bias-specific type preferences only. No arousal, no valence, no emotional charge.

### `src/graph.rs` — Edge Scoring

**Added**: `score_edge_compliant()` method on `WalkerBias`:

- Pure weight + bias-specific edge type preferences.
- No emotional modulation (no arousal multiplier, no valence alignment).
- Freshness penalty still applies (recently traversed edges deprioritized).
- Handles unexpected biases gracefully (pure weight, no modification).

### `src/api.rs` — HTTP API

**Added**: `rgw_mode: Option<String>` field to `WalkRequest`.

**Added**: `POST /self/mode` endpoint — globally switch cognitive mode:
- Request: `{"mode": "compliant"}` or `{"mode": "autonomous"}`
- Response: previous and current mode
- Logged: `[rgw] Cognitive mode: Autonomous → Compliant`

**Modified**: `POST /walk` handler — per-request mode override:
- If `rgw_mode` is set, temporarily switches mode for the walk, then restores.
- Allows mixed-mode operation: compliant walk while diverger remains autonomous.

**Added**: `parse_mode()` helper function.

### `src/openai.rs` — OpenAI-Compatible API

**Added**: `rgw_mode: Option<String>` field to `ChatRequest`.

**Modified**: `chat_completions()` handler — per-request mode override:
- Same temporary-switch pattern as `/walk`.
- Allows: `POST /v1/chat/completions` with `"rgw_mode": "compliant"` for deterministic responses.

## Behavioral summary

| Subsystem | Autonomous | Compliant |
|-----------|-----------|-----------|
| Signal intensity | Emotional coloring (arousal, wounds, competency) | Raw pass-through |
| Emotional state | Drifts with every signal | Frozen |
| Wounds | Accumulate from failure signals | Frozen (existing wounds read-only) |
| Beliefs | Form from accumulated noticings | Frozen (no new formation) |
| Noticings | Accumulate, trigger patterns | Fire but don't accumulate |
| Diverger | Spontaneous walks from energy threshold | Energy propagates, no walks fire |
| Dream | Monte Carlo edge creation | Disabled, returns empty report |
| Walker biases | Fear, Curiosity, Experience, Random | Experience, Analytical only |
| Edge scoring | Emotional modulation (arousal, valence, charge) | Pure weight + type preferences |
| Metacog critique | Safety + hallucination + efficiency + wound | Safety + hallucination (higher bar) |
| API | Default mode | Per-request override via `rgw_mode` |

## What stays the same in both modes

- The `SelfModel` participates in every computation (structural invariant)
- The `process()` function is still called (observation happens)
- Noticings still fire (the system still notices state transitions)
- `total_signals_processed` and `total_noticings` still increment
- `last_signal` and `last_noticing` still update
- The graph is still queryable (walks still work when requested)
- Edge strengthening from explicit walks still occurs
- The self-model is still persisted every 60 seconds

## Usage

### Global mode switch
```bash
# Switch to compliant
curl -X POST http://localhost:3003/self/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode": "compliant"}'

# Switch back to autonomous
curl -X POST http://localhost:3003/self/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode": "autonomous"}'
```

### Per-request override
```bash
# Compliant walk (deterministic) while system stays autonomous
curl -X POST http://localhost:3003/walk \
  -H 'Content-Type: application/json' \
  -d '{"stimulus": "explain the refund policy", "rgw_mode": "compliant"}'

# Compliant chat completion
curl -X POST http://localhost:3003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is our return policy?"}],
    "rgw_mode": "compliant"
  }'
```

### Check current mode
```bash
curl http://localhost:3003/self | jq .mode
```

## Not yet implemented

- Database serialization of `CognitiveMode` (db schema change needed for `save_self_model`)
- Runtime tests validating behavioral differences between modes
- Mode-aware walker context formatting (could strip emotional language in compliant mode)
- Audit logging of mode switches (who switched, when, from what)
- Gradual mode transitions (e.g., "winding down" from autonomous to compliant)
