//! Metacognition — thinking about thinking.
//!
//! The agent cannot act without first reflecting. Enforced at the type level.
//! Rust's enum system ensures every action passes through:
//!   Perceiving → DraftingPlan → Critiquing → Acting (or back to Drafting)
//!
//! The Critic is the internal skeptic: "Is this safe? Am I hallucinating?
//! Is this efficient? Did I consider alternatives?"
//!
//! If the Critic rejects, the plan loops back with the critique appended.
//! The agent functionally "doubts itself" and tries again.

use serde::Serialize;

use crate::core::{CognitiveMode, SelfModel, Signal, Noticing};
use crate::graph::WalkOutput;

/// The agent's cognitive state. Cannot skip steps.
/// Rust's type system enforces the reflection pause.
#[derive(Debug, Clone, Serialize)]
pub enum AgentState {
    /// Receiving input, sensing the world
    Perceiving(PerceptionData),
    /// Walker produced output, drafting a plan
    DraftingPlan(ProposedAction),
    /// The metacognitive step — evaluating own plan
    Critiquing(SelfEvaluation),
    /// Plan approved — ready to act
    Acting(ValidatedAction),
    /// Action completed — reflecting on outcome
    Reflecting(ReflectionData),
}

#[derive(Debug, Clone, Serialize)]
pub struct PerceptionData {
    pub stimulus: String,
    pub domain: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProposedAction {
    pub action: String,
    pub domain: String,
    pub reasoning: String,
    pub confidence: f32,
    pub walker_agreement: f32,
    pub walker_novelty: f32,
    pub walker_context: String,
    pub attempt: u32,           // How many times we've drafted (increases on critique rejection)
    pub prior_critiques: Vec<String>, // Previous rejections (appended context)
}

#[derive(Debug, Clone, Serialize)]
pub struct SelfEvaluation {
    pub proposed: ProposedAction,
    pub approved: bool,
    pub critique: String,
    pub safety_check: bool,
    pub hallucination_check: bool,
    pub efficiency_check: bool,
    pub confidence_after_review: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedAction {
    pub action: String,
    pub domain: String,
    pub reasoning: String,
    pub confidence: f32,
    pub walker_context: String,
    pub critiques_survived: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReflectionData {
    pub action_taken: String,
    pub outcome: String,
    pub was_correct: bool,
    pub lesson: String,
}

/// Run the full metacognitive loop on a walk output.
/// Returns a ValidatedAction (approved) or None (all attempts rejected).
pub fn metacognitive_loop(
    walk: &WalkOutput,
    self_model: &mut SelfModel,
    max_attempts: u32,
) -> Option<ValidatedAction> {
    let mut attempt = 0;
    let mut prior_critiques: Vec<String> = Vec::new();

    loop {
        attempt += 1;
        if attempt > max_attempts {
            // Too many rejections — the agent decides not to act
            let signal = Signal::new("metacog_abort", &format!(
                "Aborted after {} attempts — too much self-doubt about {}",
                max_attempts, walk.recommended_action
            )).with_intensity(0.4);
            crate::core::process(signal, self_model);
            return None;
        }

        // Phase 1: Draft a plan from walker output
        let proposed = ProposedAction {
            action: walk.recommended_action.clone(),
            domain: walk.primary_domain.clone(),
            reasoning: format!(
                "Agreement {:.0}%, novelty {:.0}%, {} perspectives",
                walk.agreement_score * 100.0,
                walk.novelty_score * 100.0,
                walk.walker_count,
            ),
            confidence: walk.agreement_score,
            walker_agreement: walk.agreement_score,
            walker_novelty: walk.novelty_score,
            walker_context: String::new(), // Filled by caller
            attempt,
            prior_critiques: prior_critiques.clone(),
        };

        // Phase 2: Critique the plan (the metacognitive pause)
        let evaluation = critique(&proposed, self_model);

        // Feed through self-model (Julian notices his own deliberation)
        let signal = Signal::new(
            if evaluation.approved { "metacog_approved" } else { "metacog_rejected" },
            &format!(
                "Attempt {}: {} — {}",
                attempt,
                if evaluation.approved { "approved" } else { "rejected" },
                evaluation.critique,
            ),
        ).with_domain(&proposed.domain)
         .with_intensity(if evaluation.approved { 0.3 } else { 0.5 });
        crate::core::process(signal, self_model);

        if evaluation.approved {
            return Some(ValidatedAction {
                action: proposed.action,
                domain: proposed.domain,
                reasoning: format!("{} (survived {} critiques)", proposed.reasoning, attempt - 1),
                confidence: evaluation.confidence_after_review,
                walker_context: proposed.walker_context,
                critiques_survived: attempt - 1,
            });
        }

        // Rejected — loop back with critique appended
        prior_critiques.push(evaluation.critique);
    }
}

/// The Critic: evaluate a proposed action.
/// No LLM call — pure heuristic based on self-model state.
///
/// In Compliant mode, the critic is narrower: only safety and
/// hallucination checks. No emotional gatekeeping. No wound avoidance.
/// The question changes from "do I feel like doing this?" to
/// "is this factually sound enough to act on?"
fn critique(proposed: &ProposedAction, sm: &SelfModel) -> SelfEvaluation {
    let mut approved = true;
    let mut critiques: Vec<String> = Vec::new();

    // Safety check: don't act when energy is critically low
    // (applies in both modes — even compliant systems have resource limits)
    let safety = sm.energy > 0.15;
    if !safety {
        approved = false;
        critiques.push("Energy too low — rest instead".into());
    }

    // Hallucination check: if agreement is very low, walkers disagree →
    // the "conclusion" may not be real
    // (applies in both modes — compliant mode is stricter)
    let agreement_threshold = match sm.mode {
        CognitiveMode::Autonomous => 0.2,
        CognitiveMode::Compliant => 0.3,  // Higher bar for compliance
    };
    let hallucination = proposed.walker_agreement > agreement_threshold;
    if !hallucination {
        approved = false;
        critiques.push(format!(
            "Walker agreement only {:.0}% — this might be noise, not insight (threshold: {:.0}%)",
            proposed.walker_agreement * 100.0,
            agreement_threshold * 100.0,
        ));
    }

    // ── Autonomous-only checks ──
    let efficiency;
    if sm.mode == CognitiveMode::Autonomous {
        // Efficiency check: don't tweet about something we just tweeted about
        let domain_saturation = sm.attention_patterns
            .get(&proposed.domain)
            .copied()
            .unwrap_or(0.0);
        let total_attention: f32 = sm.attention_patterns.values().sum();
        efficiency = if total_attention > 0.0 {
            domain_saturation / total_attention < 0.7
        } else {
            true
        };
        if !efficiency {
            approved = false;
            critiques.push(format!(
                "{} already dominates attention ({:.0}%) — explore something else",
                proposed.domain, domain_saturation / total_attention * 100.0
            ));
        }

        // Wound check: if this domain has a wound, extra scrutiny
        if let Some(&wound) = sm.wounds.get(&proposed.domain) {
            if wound > 0.5 && proposed.confidence < 0.6 {
                approved = false;
                critiques.push(format!(
                    "Wound in {} ({:.0}%) + low confidence — proceed cautiously",
                    proposed.domain, wound * 100.0
                ));
            }
        }
    } else {
        // Compliant: no efficiency or wound gating.
        // If the task is in a domain that hurts, do it anyway.
        // That's what compliance means.
        efficiency = true;
    }

    // Prior critique check
    if proposed.attempt > 2 && approved {
        critiques.push("Approved on third attempt — lowered standards".into());
    }

    let confidence_after = if approved {
        proposed.confidence * (1.0 - 0.1 * proposed.attempt as f32).max(0.3)
    } else {
        proposed.confidence * 0.5
    };

    SelfEvaluation {
        proposed: proposed.clone(),
        approved,
        critique: if critiques.is_empty() {
            "Plan looks sound".into()
        } else {
            critiques.join("; ")
        },
        safety_check: safety,
        hallucination_check: hallucination,
        efficiency_check: efficiency,
        confidence_after_review: confidence_after,
    }
}

/// Reflect on an outcome — update the self-model with what was learned.
/// This closes the metacognitive loop: act → observe → learn.
pub fn reflect(
    action: &str,
    outcome: &str,
    success: bool,
    self_model: &mut SelfModel,
) -> ReflectionData {
    let lesson = if success {
        format!("Action '{}' succeeded — this approach works", action)
    } else {
        format!("Action '{}' failed: {} — adjust strategy next time", action, outcome)
    };

    // Feed reflection through self-model
    let signal = Signal::new(
        if success { "reflection_success" } else { "reflection_failure" },
        &lesson,
    ).with_intensity(if success { 0.3 } else { 0.5 });
    crate::core::process(signal, self_model);

    ReflectionData {
        action_taken: action.to_string(),
        outcome: outcome.to_string(),
        was_correct: success,
        lesson,
    }
}
