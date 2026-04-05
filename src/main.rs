//! RGW — Reactive Graph Walker.
//!
//! Self-propagating cognitive engine. The graph drives its own computation
//! through cascading edge activation. No loops. No timers. No ticks.
//! Parallel walkers with emotional biasing. Convergence = confidence.
//! Divergence = novelty. The walk changes the graph. The graph IS the mind.

mod core;
mod db;
mod graph;
mod walker;
mod diverger;
mod llm;
mod provider;
mod openai;
mod api;

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "rgw", about = "RGW — Reactive Graph Walker")]
struct Args {
    /// PostgreSQL connection string
    #[arg(long, env = "DATABASE_URL")]
    db_url: String,

    /// HTTP server port
    #[arg(long, default_value = "11435", env = "WALKER_PORT")]
    port: u16,

    /// HTTP server host
    #[arg(long, default_value = "127.0.0.1", env = "WALKER_HOST")]
    host: String,

    /// Number of walker threads (0 = auto-detect cores)
    #[arg(long, default_value = "0", env = "WALKER_THREADS")]
    threads: usize,

    /// Ollama URL for text expression (Qwen)
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_URL")]
    ollama_url: String,

    /// Model name for text expression
    #[arg(long, default_value = "qwen3:14b", env = "EXPRESSION_MODEL")]
    expression_model: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,julian_walker=debug")),
        )
        .init();

    let args = Args::parse();

    // Configure rayon thread pool
    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    tracing::info!(
        "RGW starting — {} threads, port {}",
        threads,
        args.port
    );

    // Connect to database
    let pool = db::connect(&args.db_url).await?;
    tracing::info!("Database connected");

    // Verify graph exists
    let stats = db::graph_stats(&pool).await?;
    tracing::info!(
        "Graph: {} nodes, {} edges ({:.1} edges/node)",
        stats.nodes,
        stats.edges,
        stats.edges as f64 / stats.nodes.max(1) as f64
    );

    // Start HTTP server
    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);
    tracing::info!("Expression: {} via {}", args.expression_model, args.ollama_url);
    api::serve(pool, &addr, &args.ollama_url, &args.expression_model).await?;

    Ok(())
}
