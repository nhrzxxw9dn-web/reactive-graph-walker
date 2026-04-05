//! Graph types and emotional modulation logic.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Emotional state — modulates how walkers traverse edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f32,  // -1 to 1
    pub arousal: f32,  // 0 to 1
    pub energy: f32,   // 0 to 1
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.3,
            energy: 0.7,
        }
    }
}

/// Walker bias — each bias changes which edges a walker prefers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WalkerBias {
    Fear,
    Curiosity,
    Experience,
    Random,
    Analytical,
    Contrarian,
}

impl WalkerBias {
    /// All available biases
    pub fn all() -> &'static [WalkerBias] {
        &[
            WalkerBias::Fear,
            WalkerBias::Curiosity,
            WalkerBias::Experience,
            WalkerBias::Random,
        ]
    }

    /// Score an edge based on this bias + emotional state
    pub fn score_edge(
        &self,
        edge_type: &str,
        edge_weight: f32,
        emotional_charge: f32,
        traversal_count: i32,
        emotion: &EmotionalState,
    ) -> f32 {
        let mut w = edge_weight;

        match self {
            WalkerBias::Fear => {
                if edge_type == "caused" || edge_type == "contradicts" {
                    w *= 2.5;
                }
            }
            WalkerBias::Curiosity => {
                if edge_type == "reminds_of" {
                    w *= 1.5;
                }
                // Follow weak/unexplored edges
                if edge_weight < 0.3 {
                    w += 3.0;
                }
            }
            WalkerBias::Experience => {
                if edge_type == "reinforces" || edge_type == "similar" {
                    w *= 2.0;
                }
            }
            WalkerBias::Analytical => {
                if edge_type == "caused" || edge_type == "reinforces" {
                    w *= 2.0;
                }
            }
            WalkerBias::Contrarian => {
                if edge_type == "contradicts" {
                    w *= 3.0;
                }
            }
            WalkerBias::Random => {
                // Pure exploration — weight is random
                return rand::rng().random::<f32>();
            }
        }

        // Emotional modulation (applies to all biases except random)
        w *= 1.0 + emotion.arousal; // High arousal = wider reach

        // Valence alignment
        if emotion.valence.abs() > 0.3 {
            let alignment = 1.0 - (emotional_charge - emotion.valence).abs();
            w *= 0.5 + alignment;
        }

        // Freshness penalty — recently traversed edges slightly deprioritized
        if traversal_count > 5 {
            w *= 0.8;
        }

        w.max(0.001)
    }
}

/// Result of a single walker's traversal
#[derive(Debug, Clone, Serialize)]
pub struct WalkerResult {
    pub bias: WalkerBias,
    pub path: Vec<i32>,
    pub domains_visited: Vec<String>,
    pub edge_types_used: Vec<String>,
    pub total_weight: f32,
    pub surprises: usize,   // cross-domain transitions
    pub dead_ends: usize,
    pub edges_traversed: Vec<i32>,  // edge IDs for batch strengthening
}

/// Aggregated output of all parallel walkers
#[derive(Debug, Clone, Serialize)]
pub struct WalkOutput {
    pub recommended_action: String,
    pub primary_domain: String,
    pub domain_distribution: std::collections::HashMap<String, usize>,
    pub agreement_score: f32,
    pub novelty_score: f32,
    pub emotional_resonance: f32,
    pub search_query: Option<String>,
    pub expression_seeds: Vec<serde_json::Value>,
    pub novel_connections: usize,
    pub consensus_nodes: Vec<i32>,
    pub divergent_nodes: Vec<i32>,
    pub blind_spots: Vec<i32>,
    pub walker_count: usize,
    pub total_hops: usize,
    pub walk_ms: f64,
    pub total_ms: f64,
    pub hops_per_sec: f64,
}
