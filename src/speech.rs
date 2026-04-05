//! Speech — TTS and STT for RGW.
//!
//! TTS: text → audio (Julian speaks)
//! STT: audio → text (Julian listens)
//!
//! Backends (planned):
//!   - Apple TTS (macOS native, free, low latency)
//!   - Coqui TTS (open source, custom voice cloning)
//!   - Whisper (OpenAI STT, runs locally via whisper.cpp)
//!   - Cloud: Google TTS, ElevenLabs (higher quality, costs money)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    pub tts_engine: String,      // "apple", "coqui", "elevenlabs", "disabled"
    pub stt_engine: String,      // "whisper", "disabled"
    pub voice_id: String,        // Voice to use
    pub sample_rate: u32,        // Audio sample rate
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            tts_engine: "disabled".into(),
            stt_engine: "disabled".into(),
            voice_id: "julian_v1".into(),
            sample_rate: 24000,
        }
    }
}

/// Text-to-speech: convert text to audio bytes.
pub async fn speak(text: &str, config: &SpeechConfig) -> Result<Vec<u8>, String> {
    match config.tts_engine.as_str() {
        "apple" => speak_apple(text).await,
        "elevenlabs" => Err("ElevenLabs TTS not yet implemented".into()),
        "coqui" => Err("Coqui TTS not yet implemented".into()),
        "disabled" => Err("TTS disabled".into()),
        other => Err(format!("Unknown TTS engine: {}", other)),
    }
}

/// Speech-to-text: convert audio bytes to text.
pub async fn listen(audio: &[u8], config: &SpeechConfig) -> Result<String, String> {
    match config.stt_engine.as_str() {
        "whisper" => listen_whisper(audio).await,
        "disabled" => Err("STT disabled".into()),
        other => Err(format!("Unknown STT engine: {}", other)),
    }
}

/// Apple TTS via `say` command (macOS only, zero dependencies)
async fn speak_apple(text: &str) -> Result<Vec<u8>, String> {
    use tokio::process::Command;

    let tmp = format!("/tmp/rgw_speech_{}.aiff", std::process::id());

    let status = Command::new("say")
        .args(["-o", &tmp, text])
        .status()
        .await
        .map_err(|e| format!("say command failed: {}", e))?;

    if !status.success() {
        return Err("say command returned non-zero".into());
    }

    let audio = tokio::fs::read(&tmp)
        .await
        .map_err(|e| format!("Failed to read audio: {}", e))?;

    let _ = tokio::fs::remove_file(&tmp).await;

    Ok(audio)
}

/// Whisper STT (placeholder — needs whisper.cpp integration)
async fn listen_whisper(_audio: &[u8]) -> Result<String, String> {
    Err("Whisper STT not yet implemented — needs whisper.cpp integration".into())
}
