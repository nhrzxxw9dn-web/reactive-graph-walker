//! Julian Walker — parallel graph traversal cognitive engine.
//!
//! Replaces LLM-based decision-making with emotionally-biased graph walking.
//! Multiple walkers traverse the memory graph simultaneously on separate cores.
//! Convergence = confidence. Divergence = novelty.
//! The walk changes the graph. The graph IS the mind.

mod db;
mod graph;
mod walker;
mod diverger;
mod api;

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "julian-walker", about = "Julian's cognitive engine")]
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
        "Julian Walker starting — {} threads, port {}",
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
    api::serve(pool, &addr).await?;

    Ok(())
}
