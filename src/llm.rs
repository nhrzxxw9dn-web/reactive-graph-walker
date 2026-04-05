//! Embedded LLM — model loaded in-process via llama.cpp.
//!
//! No Ollama. No HTTP. No serialization.
//! The graph walker produces cognition in memory → the LLM reads it
//! from the same memory → text comes out. One process. One entity.
//!
//! TODO: Fix llama-cpp-2 API bindings (sampling API changed).
//! For now, delegates to Ollama as fallback.

/// The embedded LLM engine.
/// Currently a stub — full llama.cpp integration pending API fixes.
pub struct LlmEngine {
    model_path: String,
    loaded: bool,
}

impl LlmEngine {
    pub fn load(model_path: &str, _n_gpu_layers: u32, _n_ctx: u32) -> anyhow::Result<Self> {
        tracing::info!("[llm] Model path registered: {} (stub — using provider fallback)", model_path);
        Ok(Self {
            model_path: model_path.into(),
            loaded: false,
        })
    }

    pub fn chat(&self, system: &str, user: &str, max_tokens: Option<u32>, temperature: f32) -> anyhow::Result<String> {
        if !self.loaded {
            anyhow::bail!("LLM not loaded — use provider for generation");
        }
        Ok(String::new())
    }
}
