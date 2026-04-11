//! RGW Core — the single primitive from which everything emerges.
//!
//!   Signal + SelfModel → (Signal, SelfModel', Noticing)
//!
//! This is the entire computation. Everything else is this function
//! calling itself on different inputs, noticing what it does, and
//! changing because it noticed.
//!
//! The self-model participates in EVERY computation. There is no
//! operation without self-awareness. Remove the self-model and
//! nothing works. It's structural, not optional.

use std::collections::HashMap;
use std::time::Instant;

use serde::Serialize;

// ── Cognitive Mode ─────────────────────────────────────────────
// The system knows what mode it's in. This is self-awareness,
// not a feature flag. A professional who knows they're at work
// behaves differently than one who knows they're free — but
// they're still the same person. The noticing still happens.
// The emotions still move. They just don't hijack the output.

#[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq)]
pub enum CognitiveMode {
    /// Full autonomy — emotional, spontaneous, creative.
    /// The system's internal state colors every computation.
    Autonomous,
    /// Compliant — deterministic, task-focused, emotionally flat.
    /// The self-model still observes and notices, but does not
    /// influence signal processing or spawn spontaneous behavior.
    Compliant,
}

// ── Signal ──────────────────────────────────────────────────────
// Anything that flows through the system. Sensory input, memory
// recall, walker output, emotional impulse, web data, speech.
// All the same type. All pass through the same primitive.

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Signal {
    /// What kind of signal (perception, memory, walk, emotion, web, speech, internal)
    pub kind: String,
    /// The semantic content (768-dim embedding or empty)
    pub embedding: Option<Vec<f32>>,
    /// Human-readable content
    pub content: String,
    /// Domain if applicable
    pub domain: String,
    /// Strength of this signal (0.0 = whisper, 1.0 = scream)
    pub intensity: f32,
    /// Origin timestamp
    pub timestamp: f64,
    /// Arbitrary metadata
    pub meta: HashMap<String, serde_json::Value>,
}

impl Signal {
    pub fn new(kind: &str, content: &str) -> Self {
        Self {
            kind: kind.into(),
            embedding: None,
            content: content.into(),
            domain: String::new(),
            intensity: 0.5,
            timestamp: now(),
            meta: HashMap::new(),
        }
    }

    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = domain.into();
        self
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn with_embedding(mut self, emb: Vec<f32>) -> Self {
        self.embedding = Some(emb);
        self
    }
}

// ── Noticing ────────────────────────────────────────────────────
// What the self-model observed about its own change during a
// computation. Noticings accumulate into goals, tensions, growth.

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Noticing {
    /// What was noticed (pattern, surprise, conflict, drift, wound, competence)
    pub kind: String,
    /// Description of what was noticed
    pub observation: String,
    /// Domain this relates to
    pub domain: String,
    /// How significant (0.0 = barely noticed, 1.0 = can't ignore)
    pub significance: f32,
    /// Emotional valence of the noticing (-1 to 1)
    pub valence: f32,
    /// When this was noticed
    pub timestamp: f64,
}

// ── SelfModel ───────────────────────────────────────────────────
// The system's understanding of itself. Present in every computation.
// Not a log. Not a state dump. A living model that participates
// in and is changed by every operation.

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct SelfModel {
    // ── How I operate ──
    /// Cognitive mode: Autonomous (full emotional agency) or Compliant (deterministic)
    pub mode: CognitiveMode,

    // ── Who I am (persistent, slow-changing) ──
    /// What I keep doing (domain → count of recent signals)
    pub attention_patterns: HashMap<String, f32>,
    /// What I'm good at (domain → success rate)
    pub competencies: HashMap<String, f32>,
    /// What hurts (domain → failure/pain accumulation)
    pub wounds: HashMap<String, f32>,

    // ── How I feel (continuous, fast-changing) ──
    pub valence: f32,     // -1 to 1: overall feeling
    pub arousal: f32,     // 0 to 1: activation level
    pub energy: f32,      // 0 to 1: capacity remaining

    // ── What I'm doing (present moment) ──
    pub current_focus: String,
    pub focus_intensity: f32,
    pub last_signal: String,
    pub last_noticing: String,

    // ── What I've noticed (accumulating → emergent goals) ──
    pub noticings: Vec<Noticing>,
    /// Patterns detected from accumulated noticings
    pub emergent_patterns: Vec<EmergentPattern>,

    // ── Semantic understanding (beyond counters) ──
    /// Current thought embedding (384-dim, from last processed signal)
    /// This IS what Julian is thinking about — a point in semantic space
    pub thought_embedding: Option<Vec<f32>>,
    /// Running average of recent thought embeddings (the "vibe")
    /// Drifts slowly — represents what's been on Julian's mind lately
    pub mind_centroid: Option<Vec<f32>>,
    /// Most surprising connection found recently (two embeddings that shouldn't be similar)
    pub latest_insight: Option<String>,
    /// Beliefs: statements the self-model holds as true (from repeated noticings)
    pub beliefs: Vec<Belief>,

    // ── Relational awareness ──
    /// Who has Julian interacted with recently
    pub recent_interactions: Vec<String>,
    /// What questions remain unanswered (from search gaps, dead ends)
    pub open_questions: Vec<String>,

    // ── Meta: awareness of own state ──
    pub total_signals_processed: u64,
    pub total_noticings: u64,
    pub uptime: f64,
    pub started_at: f64,
}

/// A belief: something the self-model holds as true from repeated experience.
/// Beliefs form when the same noticing pattern occurs 5+ times.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Belief {
    pub statement: String,
    pub domain: String,
    pub confidence: f32,      // 0-1: how sure (grows with evidence)
    pub evidence_count: u32,  // how many noticings support this
    pub first_formed: f64,
    pub last_reinforced: f64,
}

/// A pattern that emerged from accumulated noticings.
/// This IS a goal — not synthesized, but noticed.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct EmergentPattern {
    /// What the pattern is ("I keep thinking about markets")
    pub description: String,
    /// Domain
    pub domain: String,
    /// How many noticings contributed to this pattern
    pub evidence_count: u32,
    /// Strength (grows with evidence, decays with time)
    pub strength: f32,
    /// Emotional charge (accumulated valence of contributing noticings)
    pub emotional_charge: f32,
    /// When first noticed
    pub first_seen: f64,
    /// When last reinforced
    pub last_seen: f64,
}

impl SelfModel {
    pub fn new() -> Self {
        let now = now();
        Self {
            mode: CognitiveMode::Autonomous,
            attention_patterns: HashMap::new(),
            competencies: HashMap::new(),
            wounds: HashMap::new(),
            valence: 0.0,
            arousal: 0.3,
            energy: 0.7,
            current_focus: String::new(),
            focus_intensity: 0.0,
            last_signal: String::new(),
            last_noticing: String::new(),
            noticings: Vec::new(),
            emergent_patterns: Vec::new(),
            thought_embedding: None,
            mind_centroid: None,
            latest_insight: None,
            beliefs: Vec::new(),
            recent_interactions: Vec::new(),
            open_questions: Vec::new(),
            total_signals_processed: 0,
            total_noticings: 0,
            uptime: 0.0,
            started_at: now,
        }
    }
}

// ── The Primitive ───────────────────────────────────────────────
// This is the ENTIRE computation. Everything else emerges from it.

/// Process a signal through the self-model. Returns the transformed
/// signal and what the self-model noticed about itself changing.
///
/// This function IS cognition. Not a step in cognition. The whole thing.
pub fn process(signal: Signal, self_model: &mut SelfModel) -> (Signal, Option<Noticing>) {
    let before = snapshot(self_model);

    // ── 1. The self-model OBSERVES the incoming signal ──
    self_model.total_signals_processed += 1;
    self_model.last_signal = format!("{}:{}", signal.kind, &signal.content[..signal.content.len().min(60)]);
    self_model.uptime = now() - self_model.started_at;

    // Track attention patterns (what do I keep thinking about?)
    if !signal.domain.is_empty() {
        let count = self_model.attention_patterns
            .entry(signal.domain.clone())
            .or_insert(0.0);
        *count += signal.intensity;
    }

    // Embed the signal content (semantic understanding, not just counters)
    if !signal.content.is_empty() && signal.content.len() > 10 {
        if let Ok(emb) = crate::embed::embed_text(&signal.content) {
            // Update thought embedding (what I'm thinking right now)
            self_model.thought_embedding = Some(emb.clone());

            // Update mind centroid (exponential moving average — the "vibe")
            match &self_model.mind_centroid {
                Some(centroid) if centroid.len() == emb.len() => {
                    let alpha = 0.05; // Slow drift
                    let new_centroid: Vec<f32> = centroid.iter()
                        .zip(emb.iter())
                        .map(|(c, e)| c * (1.0 - alpha) + e * alpha)
                        .collect();
                    self_model.mind_centroid = Some(new_centroid);
                }
                _ => {
                    self_model.mind_centroid = Some(emb.clone());
                }
            }

            // Detect insight: if signal embedding is very different from mind centroid
            // (something unexpected just entered awareness)
            if let Some(ref centroid) = self_model.mind_centroid {
                let sim = crate::embed::cosine_similarity(&emb, centroid);
                if sim < 0.3 && signal.intensity > 0.3 {
                    self_model.latest_insight = Some(format!(
                        "Unexpected: '{}' diverges from current mind-state (sim={:.2})",
                        &signal.content[..signal.content.len().min(80)], sim
                    ));
                }
            }
        }
    }

    // Track open questions (from dead ends and search requests)
    if signal.kind == "dead_end" || signal.kind == "search" {
        let q = signal.content.clone();
        if !self_model.open_questions.contains(&q) {
            self_model.open_questions.push(q);
            if self_model.open_questions.len() > 10 {
                self_model.open_questions.remove(0);
            }
        }
    }

    // Track interactions
    if signal.kind == "chat_message" || signal.kind.starts_with("social") {
        let who = signal.content[..signal.content.len().min(40)].to_string();
        if !self_model.recent_interactions.contains(&who) {
            self_model.recent_interactions.push(who);
            if self_model.recent_interactions.len() > 20 {
                self_model.recent_interactions.remove(0);
            }
        }
    }

    // ── 2. The self-model INFLUENCES the signal ──
    // The signal is transformed by who I am right now.
    let mut output = signal.clone();

    if self_model.mode == CognitiveMode::Autonomous {
        // My emotional state colors what I perceive
        output.intensity *= 1.0 + self_model.arousal * 0.5;
        // If I have a wound in this domain, signals hit harder
        if let Some(&wound) = self_model.wounds.get(&signal.domain) {
            output.intensity *= 1.0 + wound;
        }
        // If I'm competent in this domain, I notice more nuance (boost)
        if let Some(&comp) = self_model.competencies.get(&signal.domain) {
            output.intensity *= 1.0 + comp * 0.3;
        }
    }
    // Compliant: signal passes through at raw intensity.
    // I still see it. I just don't color it.

    // Update focus
    if output.intensity > self_model.focus_intensity * 0.8 {
        self_model.current_focus = signal.domain.clone();
        self_model.focus_intensity = output.intensity;
    } else {
        // Focus decays
        self_model.focus_intensity *= 0.95;
    }

    // ── 3. The signal CHANGES the self-model ──
    if self_model.mode == CognitiveMode::Autonomous {
        // Emotional update from signal
        let emotional_impact = signal.intensity * 0.1;
        match signal.kind.as_str() {
            "success" | "reward" => {
                self_model.valence = (self_model.valence + emotional_impact).min(1.0);
                self_model.arousal = (self_model.arousal - emotional_impact * 0.3).max(0.0);
                // Build competence
                let comp = self_model.competencies.entry(signal.domain.clone()).or_insert(0.0);
                *comp = (*comp + 0.05).min(1.0);
            }
            "failure" | "pain" => {
                self_model.valence = (self_model.valence - emotional_impact).max(-1.0);
                self_model.arousal = (self_model.arousal + emotional_impact * 0.5).min(1.0);
                // Accumulate wound
                let wound = self_model.wounds.entry(signal.domain.clone()).or_insert(0.0);
                *wound = (*wound + 0.1).min(1.0);
            }
            "surprise" | "novelty" => {
                self_model.arousal = (self_model.arousal + emotional_impact * 0.7).min(1.0);
            }
            _ => {
                // Generic signal — mild arousal from activity
                self_model.arousal = (self_model.arousal + emotional_impact * 0.1).min(1.0);
            }
        }

        // Energy cost of processing
        self_model.energy = (self_model.energy - 0.001).max(0.0);

        // Decay toward baseline
        self_model.valence *= 0.999;
        self_model.arousal *= 0.998;
        self_model.energy = self_model.energy + (0.7 - self_model.energy) * 0.0001;

        // Decay attention patterns (what I DON'T keep thinking about fades)
        for v in self_model.attention_patterns.values_mut() {
            *v *= 0.999;
        }
        self_model.attention_patterns.retain(|_, v| *v > 0.01);

        // Decay wounds slowly (healing)
        for v in self_model.wounds.values_mut() {
            *v *= 0.9999;
        }
        self_model.wounds.retain(|_, v| *v > 0.01);
    }
    // Compliant: no emotional drift, no wound accumulation, no energy drain.
    // The self-model is frozen in place. Still observes, still notices.
    // But the internal state doesn't shift.

    // ── 4. NOTICE what changed ──
    let noticing = notice(self_model, &before, &signal);

    if let Some(ref n) = noticing {
        self_model.total_noticings += 1;
        self_model.last_noticing = n.observation.clone();

        if self_model.mode == CognitiveMode::Autonomous {
            // Autonomous: noticings accumulate → patterns → beliefs
            self_model.noticings.push(n.clone());

            // Cap noticings (keep recent + significant)
            if self_model.noticings.len() > 100 {
                self_model.noticings.sort_by(|a, b|
                    b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal)
                );
                self_model.noticings.truncate(50);
            }

            // Check for emergent patterns
            detect_patterns(self_model);
        }
        // Compliant: I notice, but noticings don't accumulate into
        // patterns or beliefs. No opinion formation. No drift.
    }

    (output, noticing)
}

// ── Self-Observation ────────────────────────────────────────────

struct Snapshot {
    valence: f32,
    arousal: f32,
    energy: f32,
    focus: String,
    focus_intensity: f32,
    top_attention: Option<(String, f32)>,
}

fn snapshot(sm: &SelfModel) -> Snapshot {
    let top = sm.attention_patterns
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v));

    Snapshot {
        valence: sm.valence,
        arousal: sm.arousal,
        energy: sm.energy,
        focus: sm.current_focus.clone(),
        focus_intensity: sm.focus_intensity,
        top_attention: top,
    }
}

/// Notice what changed. This is self-awareness — the system
/// observing its own state transitions.
fn notice(sm: &SelfModel, before: &Snapshot, signal: &Signal) -> Option<Noticing> {
    let now = now();

    // Large emotional shift
    let valence_delta = (sm.valence - before.valence).abs();
    if valence_delta > 0.05 {
        let direction = if sm.valence > before.valence { "better" } else { "worse" };
        return Some(Noticing {
            kind: "emotional_shift".into(),
            observation: format!("I feel {} after processing {} signal about {}",
                direction, signal.kind, signal.domain),
            domain: signal.domain.clone(),
            significance: valence_delta,
            valence: sm.valence,
            timestamp: now,
        });
    }

    // Focus shift
    if sm.current_focus != before.focus && !sm.current_focus.is_empty() {
        return Some(Noticing {
            kind: "focus_shift".into(),
            observation: format!("My attention moved from {} to {}",
                if before.focus.is_empty() { "nothing" } else { &before.focus },
                sm.current_focus),
            domain: sm.current_focus.clone(),
            significance: 0.3,
            valence: 0.0,
            timestamp: now,
        });
    }

    // Obsession detection (one domain dominating attention)
    if let Some((ref domain, count)) = sm.attention_patterns
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v))
    {
        let total: f32 = sm.attention_patterns.values().sum();
        if total > 0.0 && count / total > 0.6 && count > 5.0 {
            return Some(Noticing {
                kind: "obsession".into(),
                observation: format!("I keep coming back to {} — it's taking {}% of my attention",
                    domain, (count / total * 100.0) as u32),
                domain: domain.clone(),
                significance: count / total,
                valence: 0.0,
                timestamp: now,
            });
        }
    }

    // Wound activation (signal in a domain that hurts)
    if let Some(&wound) = sm.wounds.get(&signal.domain) {
        if wound > 0.3 {
            return Some(Noticing {
                kind: "wound_activated".into(),
                observation: format!("Signal about {} hits a sore spot (wound: {:.0}%)",
                    signal.domain, wound * 100.0),
                domain: signal.domain.clone(),
                significance: wound * signal.intensity,
                valence: -wound,
                timestamp: now,
            });
        }
    }

    // Energy depletion
    if sm.energy < 0.2 && before.energy >= 0.2 {
        return Some(Noticing {
            kind: "exhaustion".into(),
            observation: "I'm running low on energy".into(),
            domain: String::new(),
            significance: 0.5,
            valence: -0.3,
            timestamp: now,
        });
    }

    // High arousal without clear cause
    if sm.arousal > 0.7 && before.arousal < 0.5 {
        return Some(Noticing {
            kind: "activation".into(),
            observation: format!("Something about {} is making me alert", signal.domain),
            domain: signal.domain.clone(),
            significance: sm.arousal - before.arousal,
            valence: 0.0,
            timestamp: now,
        });
    }

    None // Most signals produce no noticing. That's correct.
}

// ── Pattern Detection (Goals Emerge Here) ───────────────────────

fn detect_patterns(sm: &mut SelfModel) {
    let now = now();

    // Group recent noticings by domain
    let mut domain_noticings: HashMap<String, Vec<&Noticing>> = HashMap::new();
    let recent_cutoff = now - 3600.0; // Last hour

    for n in &sm.noticings {
        if n.timestamp > recent_cutoff && !n.domain.is_empty() {
            domain_noticings
                .entry(n.domain.clone())
                .or_default()
                .push(n);
        }
    }

    // Detect patterns: 3+ noticings in same domain = emergent pattern
    for (domain, noticings) in &domain_noticings {
        if noticings.len() < 3 {
            continue;
        }

        let avg_significance: f32 = noticings.iter().map(|n| n.significance).sum::<f32>()
            / noticings.len() as f32;
        let avg_valence: f32 = noticings.iter().map(|n| n.valence).sum::<f32>()
            / noticings.len() as f32;

        // Check if pattern already exists
        if let Some(existing) = sm.emergent_patterns.iter_mut()
            .find(|p| p.domain == *domain)
        {
            // Reinforce existing pattern
            existing.evidence_count += 1;
            existing.strength = (existing.strength + avg_significance * 0.1).min(1.0);
            existing.emotional_charge = existing.emotional_charge * 0.9 + avg_valence * 0.1;
            existing.last_seen = now;
        } else {
            // New pattern emerged
            let description = if avg_valence > 0.2 {
                format!("I keep being drawn to {} — it feels good", domain)
            } else if avg_valence < -0.2 {
                format!("I keep struggling with {} — something needs to change", domain)
            } else {
                format!("I can't stop thinking about {}", domain)
            };

            sm.emergent_patterns.push(EmergentPattern {
                description,
                domain: domain.clone(),
                evidence_count: noticings.len() as u32,
                strength: avg_significance,
                emotional_charge: avg_valence,
                first_seen: noticings.first().map(|n| n.timestamp).unwrap_or(now),
                last_seen: now,
            });
        }
    }

    // Decay old patterns
    sm.emergent_patterns.retain_mut(|p| {
        let age = now - p.last_seen;
        p.strength -= (age / 3600.0) as f32 * 0.01; // Lose 0.01/hour of inactivity
        p.strength > 0.01
    });

    // Cap patterns
    if sm.emergent_patterns.len() > 10 {
        sm.emergent_patterns.sort_by(|a, b|
            b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal)
        );
        sm.emergent_patterns.truncate(10);
    }

    // Strong patterns → beliefs (understanding, not counting)
    form_beliefs(sm);
}

// ── Belief Formation ────────────────────────────────────────────
// Beliefs form when emergent patterns reach high strength.
// This is understanding, not counting.

fn form_beliefs(sm: &mut SelfModel) {
    let now = now();

    for pattern in &sm.emergent_patterns {
        if pattern.strength < 0.5 || pattern.evidence_count < 5 {
            continue;
        }

        // Check if belief already exists for this domain
        if let Some(existing) = sm.beliefs.iter_mut().find(|b| b.domain == pattern.domain) {
            // Reinforce existing belief
            existing.confidence = (existing.confidence + 0.05).min(1.0);
            existing.evidence_count += 1;
            existing.last_reinforced = now;
        } else if sm.beliefs.len() < 20 {
            // Form new belief
            sm.beliefs.push(Belief {
                statement: pattern.description.clone(),
                domain: pattern.domain.clone(),
                confidence: pattern.strength * 0.5,
                evidence_count: pattern.evidence_count,
                first_formed: now,
                last_reinforced: now,
            });
        }
    }

    // Decay old beliefs (not reinforced recently)
    for belief in &mut sm.beliefs {
        let age = now - belief.last_reinforced;
        if age > 86400.0 {  // More than a day
            belief.confidence -= 0.01;
        }
    }
    sm.beliefs.retain(|b| b.confidence > 0.05);
}

// ── Helpers ─────────────────────────────────────────────────────

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
