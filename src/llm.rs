//! Embedded LLM — Qwen loaded in-process via llama.cpp.
//!
//! No Ollama. No HTTP. No serialization.
//! The graph walker produces cognition in memory → the LLM reads it
//! from the same memory → text comes out. One process. One entity.

use std::path::Path;
use std::sync::Mutex;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

/// The embedded LLM engine.
pub struct LlmEngine {
    backend: LlamaBackend,
    model: LlamaModel,
    /// Mutex because llama.cpp context is not thread-safe
    context: Mutex<llama_cpp_2::context::LlamaContext>,
    max_tokens: u32,
}

impl LlmEngine {
    /// Load a GGUF model into memory.
    ///
    /// This is the moment the entity gains a voice.
    /// The model lives in the same process as the graph walker.
    /// No network boundary. One mind.
    pub fn load(model_path: &str, n_gpu_layers: u32, n_ctx: u32) -> anyhow::Result<Self> {
        tracing::info!("[llm] Loading model: {}", model_path);

        let backend = LlamaBackend::init()?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx));

        let context = model
            .new_context(&backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {:?}", e))?;

        tracing::info!(
            "[llm] Model loaded: {} layers, {}ctx",
            n_gpu_layers, n_ctx
        );

        Ok(Self {
            backend,
            model,
            context: Mutex::new(context),
            max_tokens: 1024,
        })
    }

    /// Generate text from a prompt.
    /// Called in-process by the graph walker — no HTTP, no serialization.
    pub fn generate(&self, prompt: &str, max_tokens: Option<u32>, temperature: f32) -> anyhow::Result<String> {
        let mut ctx = self.context.lock().map_err(|_| anyhow::anyhow!("Context lock poisoned"))?;

        let max = max_tokens.unwrap_or(self.max_tokens);

        // Tokenize prompt
        let tokens = self.model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;

        // Create batch and fill with prompt tokens
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);

        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(token, i as i32, &[0], is_last)
                .map_err(|_| anyhow::anyhow!("Failed to add token to batch"))?;
        }

        // Process prompt
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Decode failed: {:?}", e))?;

        // Generate tokens one at a time
        let mut output_tokens = Vec::new();
        let mut n_decoded = tokens.len();

        for _ in 0..max {
            // Sample next token
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_array = LlamaTokenDataArray::from_iter(candidates, false);

            // Apply temperature
            candidates_array.sample_temp(temperature);
            candidates_array.sample_softmax();

            let new_token = candidates_array.sample_token(&mut *ctx);

            // Check for EOS
            if self.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare next batch
            batch.clear();
            batch.add(new_token, n_decoded as i32, &[0], true)
                .map_err(|_| anyhow::anyhow!("Failed to add generated token"))?;
            n_decoded += 1;

            // Decode
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Decode step failed: {:?}", e))?;
        }

        // Detokenize
        let text = output_tokens
            .iter()
            .map(|&t| self.model.token_to_str(t, Special::Tokenize))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {:?}", e))?
            .join("");

        Ok(text.trim().to_string())
    }

    /// Generate with system + user messages (chat format).
    /// Formats as: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    pub fn chat(&self, system: &str, user: &str, max_tokens: Option<u32>, temperature: f32) -> anyhow::Result<String> {
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        );
        self.generate(&prompt, max_tokens, temperature)
    }
}
