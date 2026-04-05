//! The walker engine — parallel graph traversal with emotional biasing.
//!
//! Each walker traverses the graph independently on its own thread (via rayon).
//! The walk changes the graph: traversed edges get strengthened.
//! Multiple walkers with different biases produce convergence/divergence signals.

use std::collections::HashMap;
use std::time::Instant;

use rand::Rng;
use rayon::prelude::*;
use sqlx::PgPool;

use crate::db;
use crate::graph::*;

/// Run a single walker through the graph.
pub fn walk_single(
    pool: &PgPool,
    seed_id: i32,
    bias: WalkerBias,
    emotion: &EmotionalState,
    steps: usize,
    rt: &tokio::runtime::Handle,
) -> WalkerResult {
    let mut result = WalkerResult {
        bias,
        path: vec![seed_id],
        domains_visited: Vec::new(),
        edge_types_used: Vec::new(),
        total_weight: 0.0,
        surprises: 0,
        dead_ends: 0,
        edges_traversed: Vec::new(),
    };

    let mut current_id = seed_id;
    let mut rng = rand::rng();
    let mut prev_domain = String::new();

    for _ in 0..steps {
        // Get edges from current node (blocking call from sync rayon thread)
        let edges = match rt.block_on(db::edges_from(pool, current_id)) {
            Ok(e) => e,
            Err(_) => {
                result.dead_ends += 1;
                break;
            }
        };

        if edges.is_empty() {
            result.dead_ends += 1;
            break;
        }

        // Score each edge based on bias + emotional state
        let scored: Vec<(&db::MemoryEdge, f32)> = edges
            .iter()
            .map(|e| {
                let score = bias.score_edge(
                    &e.edge_type,
                    e.weight,
                    e.emotional_charge,
                    e.traversal_count,
                    emotion,
                );
                (e, score)
            })
            .collect();

        // Weighted random selection (not argmax — allows surprise)
        let total: f32 = scored.iter().map(|(_, s)| s).sum();
        if total < f32::EPSILON {
            result.dead_ends += 1;
            break;
        }

        let threshold = rng.random::<f32>() * total;
        let mut cumulative = 0.0;
        let mut chosen = &scored[0];
        for item in &scored {
            cumulative += item.1;
            if cumulative >= threshold {
                chosen = item;
                break;
            }
        }

        let edge = chosen.0;

        // Determine next node (bidirectional traversal)
        let next_id = if edge.source_id == current_id {
            edge.target_id
        } else {
            edge.source_id
        };

        // Record traversal
        result.path.push(next_id);
        result.edge_types_used.push(edge.edge_type.clone());
        result.total_weight += edge.weight;
        result.edges_traversed.push(edge.id);

        // Track domain transitions
        if let Ok(Some(node)) = rt.block_on(db::get_node(pool, next_id)) {
            if !node.domain.is_empty() {
                if !prev_domain.is_empty() && prev_domain != node.domain {
                    result.surprises += 1;
                }
                if !result.domains_visited.contains(&node.domain) {
                    result.domains_visited.push(node.domain.clone());
                }
                prev_domain = node.domain;
            }
        }

        current_id = next_id;
    }

    result
}

/// Run multiple walkers in parallel and aggregate results.
pub async fn walk_parallel(
    pool: &PgPool,
    emotion: &EmotionalState,
    n_walkers: usize,
    steps: usize,
) -> WalkOutput {
    let start = Instant::now();

    // Get seed nodes
    let seeds = db::seed_nodes(pool, (n_walkers * 2) as i32)
        .await
        .unwrap_or_default();

    if seeds.is_empty() {
        return empty_output();
    }

    // Assign biases — rotate through available biases
    let biases = WalkerBias::all();
    let configs: Vec<(i32, WalkerBias)> = (0..n_walkers)
        .map(|i| {
            let seed = seeds[i % seeds.len()];
            let bias = biases[i % biases.len()];
            (seed, bias)
        })
        .collect();

    // Clone what we need for the rayon threads
    let pool_clone = pool.clone();
    let emotion_clone = emotion.clone();
    let rt = tokio::runtime::Handle::current();

    // Run walkers in parallel via rayon
    let walk_start = Instant::now();
    let results: Vec<WalkerResult> = configs
        .par_iter()
        .map(|(seed, bias)| {
            walk_single(&pool_clone, *seed, *bias, &emotion_clone, steps, &rt)
        })
        .collect();
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    // Aggregate results
    let output = aggregate(pool, results, walk_ms, start).await;

    // Batch strengthen traversed edges (the walk changes the graph)
    let all_edge_ids: Vec<i32> = output
        .consensus_nodes
        .iter()
        .copied()
        .chain(output.divergent_nodes.iter().copied())
        .collect();
    // Note: we use consensus_nodes as a proxy — actual edge strengthening
    // happens via the edge IDs collected during walks
    // For now, just log that we would strengthen

    output
}

/// Aggregate parallel walker results into structured cognition.
async fn aggregate(
    pool: &PgPool,
    results: Vec<WalkerResult>,
    walk_ms: f64,
    start: Instant,
) -> WalkOutput {
    let n = results.len();
    if n == 0 {
        return empty_output();
    }

    // Vote counting: which nodes did walkers visit?
    let mut node_votes: HashMap<i32, usize> = HashMap::new();
    let mut total_hops = 0;
    for r in &results {
        for &node_id in &r.path {
            *node_votes.entry(node_id).or_insert(0) += 1;
        }
        total_hops += r.path.len();
    }

    // Classify nodes by agreement level
    let consensus: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v as f32 > n as f32 * 0.6)
        .map(|(k, _)| *k)
        .collect();
    let divergent: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v > 1 && (**v as f32) <= n as f32 * 0.4)
        .map(|(k, _)| *k)
        .collect();
    let blind_spots: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v == 1)
        .map(|(k, _)| *k)
        .collect();

    // Domain distribution
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    for r in &results {
        for d in &r.domains_visited {
            *domain_counts.entry(d.clone()).or_insert(0) += 1;
        }
    }
    let primary_domain = domain_counts
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k.clone())
        .unwrap_or_default();

    // Agreement & novelty scores
    let total_unique = node_votes.len().max(1);
    let agreement = consensus.len() as f32 / total_unique as f32;
    let total_surprises: usize = results.iter().map(|r| r.surprises).sum();
    let novelty = ((divergent.len() + total_surprises) as f32 / total_unique as f32).min(1.0);

    // Novel connections count
    let novel_connections = total_surprises;

    // Recommended action
    let action = recommend_action(agreement, novelty, &results);

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let hops_per_sec = if total_ms > 0.0 {
        total_hops as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };

    // Build expression seeds from consensus nodes
    let mut expression_seeds = Vec::new();
    for &nid in consensus.iter().take(5) {
        if let Ok(Some(node)) = db::get_node(pool, nid).await {
            expression_seeds.push(serde_json::json!({
                "id": node.id,
                "domain": node.domain,
                "votes": node_votes.get(&nid).unwrap_or(&0),
            }));
        }
    }

    tracing::info!(
        "[walker-perf] {} walkers × {} hops = {} total | \
         walk={:.0}ms total={:.0}ms | {:.0} hops/s | \
         agreement={:.0}% novelty={:.0}%",
        n,
        total_hops / n.max(1),
        total_hops,
        walk_ms,
        total_ms,
        hops_per_sec,
        agreement * 100.0,
        novelty * 100.0,
    );

    WalkOutput {
        recommended_action: action,
        primary_domain,
        domain_distribution: domain_counts,
        agreement_score: agreement,
        novelty_score: novelty,
        emotional_resonance: 0.0,
        search_query: None,
        expression_seeds,
        novel_connections,
        consensus_nodes: consensus,
        divergent_nodes: divergent,
        blind_spots,
        walker_count: n,
        total_hops,
        walk_ms,
        total_ms,
        hops_per_sec,
    }
}

fn recommend_action(agreement: f32, novelty: f32, results: &[WalkerResult]) -> String {
    let total_surprises: usize = results.iter().map(|r| r.surprises).sum();

    if agreement > 0.6 {
        "express".to_string()
    } else if novelty > 0.6 {
        "explore".to_string()
    } else if total_surprises >= 3 {
        "search".to_string()
    } else {
        "rest".to_string()
    }
}

/// Format walk output as context for LLM prompt injection.
pub fn format_walk_context(output: &WalkOutput) -> String {
    let mut lines = Vec::new();
    lines.push("=== RGW COGNITION ===".to_string());
    lines.push(format!("Action: {}", output.recommended_action));
    lines.push(format!("Domain: {}", output.primary_domain));
    lines.push(format!("Confidence: {:.0}%", output.agreement_score * 100.0));
    lines.push(format!("Novelty: {:.0}%", output.novelty_score * 100.0));

    if !output.domain_distribution.is_empty() {
        let mut sorted: Vec<_> = output.domain_distribution.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> = sorted.iter().take(5).map(|(d, c)| format!("{}({})", d, c)).collect();
        lines.push(format!("Landscape: {}", top.join(", ")));
    }

    if !output.expression_seeds.is_empty() {
        lines.push("\nRelevant:".to_string());
        for seed in output.expression_seeds.iter().take(5) {
            let domain = seed.get("domain").and_then(|v| v.as_str()).unwrap_or("?");
            lines.push(format!("  * [{}]", domain));
        }
    }

    if output.novel_connections > 0 {
        lines.push(format!("\nNovel connections: {}", output.novel_connections));
    }

    lines.push(format!(
        "\n[{} walkers, {} hops, {:.0}ms, {:.0} hops/s]",
        output.walker_count, output.total_hops, output.total_ms, output.hops_per_sec
    ));
    lines.push("=== END RGW ===".to_string());
    lines.join("\n")
}

fn empty_output() -> WalkOutput {
    WalkOutput {
        recommended_action: "rest".to_string(),
        primary_domain: String::new(),
        domain_distribution: HashMap::new(),
        agreement_score: 0.0,
        novelty_score: 0.0,
        emotional_resonance: 0.0,
        search_query: None,
        expression_seeds: Vec::new(),
        novel_connections: 0,
        consensus_nodes: Vec::new(),
        divergent_nodes: Vec::new(),
        blind_spots: Vec::new(),
        walker_count: 0,
        total_hops: 0,
        walk_ms: 0.0,
        total_ms: 0.0,
        hops_per_sec: 0.0,
    }
}
