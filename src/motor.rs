//! Motor Cortex — RGW commands Julian's Python backend to act.
//!
//! When the Diverger fires a walk and the result says "express" or
//! "explore" or "search", the motor cortex translates that into an
//! HTTP command to Julian's Python API.
//!
//! Rust brain → Python body. The brain decides, the body executes.

use serde::Serialize;

/// Command that RGW sends to Julian's Python backend.
#[derive(Debug, Clone, Serialize)]
pub struct MotorCommand {
    /// What to do: "tweet", "blog", "journal", "search", "trade", "nothing"
    pub action: String,
    /// Which domain this relates to
    pub domain: String,
    /// Walker context (enriched prompt for the LLM)
    pub walker_context: String,
    /// Expression seeds (memory fragments to express)
    pub expression_seeds: Vec<serde_json::Value>,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Novelty (0-1)
    pub novelty: f32,
    /// Search query (if action = "search")
    pub search_query: Option<String>,
}

/// Send a motor command to Julian's Python backend.
/// Fire-and-forget: never blocks the Diverger.
pub async fn execute(julian_url: &str, command: MotorCommand) {
    if julian_url.is_empty() {
        return;
    }

    let url = format!("{}/api/admin/rgw/execute", julian_url);

    match reqwest::Client::new()
        .post(&url)
        .json(&command)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                tracing::info!(
                    "[motor] Sent: {} (domain={}, confidence={:.0}%)",
                    command.action, command.domain, command.confidence * 100.0
                );
            } else {
                tracing::warn!("[motor] Julian returned {}", resp.status());
            }
        }
        Err(e) => {
            tracing::debug!("[motor] Failed to reach Julian: {}", e);
        }
    }
}
