//! Database connection and queries for the unified graph model.

use sqlx::{postgres::PgPoolOptions, PgPool, Row};

/// Graph statistics
pub struct GraphStats {
    pub nodes: i64,
    pub edges: i64,
}

/// A memory node in the graph
#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: i32,
    pub domain: String,
    pub valence: f32,
    pub arousal: f32,
    pub importance: f32,
    pub access_count: i32,
    pub embedding: Option<Vec<f32>>,
}

/// An edge between two memory nodes
#[derive(Debug, Clone)]
pub struct MemoryEdge {
    pub id: i32,
    pub source_id: i32,
    pub target_id: i32,
    pub edge_type: String,
    pub weight: f32,
    pub emotional_charge: f32,
    pub traversal_count: i32,
}

/// Connect to PostgreSQL
pub async fn connect(url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(16)
        .connect(url)
        .await
}

/// Get graph statistics
pub async fn graph_stats(pool: &PgPool) -> Result<GraphStats, sqlx::Error> {
    let nodes: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM memory_vectors")
        .fetch_one(pool)
        .await?;

    let edges: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM memory_edges")
        .fetch_optional(pool)
        .await?
        .unwrap_or(Some(0));

    Ok(GraphStats {
        nodes: nodes.unwrap_or(0),
        edges: edges.unwrap_or(0),
    })
}

/// Get edges from a node (outgoing + incoming treated as bidirectional)
pub async fn edges_from(pool: &PgPool, node_id: i32) -> Result<Vec<MemoryEdge>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT id, source_id, target_id, edge_type, \
         COALESCE(weight, 0.5) as weight, COALESCE(emotional_charge, 0.0) as emotional_charge, \
         COALESCE(traversal_count, 0) as traversal_count \
         FROM memory_edges \
         WHERE source_id = $1 OR target_id = $1 \
         ORDER BY weight DESC \
         LIMIT 20"
    )
    .bind(node_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(|r| MemoryEdge {
        id: r.get("id"),
        source_id: r.get("source_id"),
        target_id: r.get("target_id"),
        edge_type: r.get("edge_type"),
        weight: r.get("weight"),
        emotional_charge: r.get("emotional_charge"),
        traversal_count: r.get("traversal_count"),
    }).collect())
}

/// Get a node by ID (lightweight — no embedding)
pub async fn get_node(pool: &PgPool, id: i32) -> Result<Option<MemoryNode>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT id, COALESCE(domain, '') as domain, \
         COALESCE(valence, 0.0) as valence, COALESCE(arousal, 0.5) as arousal, \
         COALESCE(importance, 5.0) as importance, COALESCE(access_count, 0) as access_count \
         FROM memory_vectors WHERE id = $1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| MemoryNode {
        id: r.get("id"),
        domain: r.get("domain"),
        valence: r.get("valence"),
        arousal: r.get("arousal"),
        importance: r.get("importance"),
        access_count: r.get("access_count"),
        embedding: None,
    }))
}

/// Get seed nodes — high importance, recently accessed
pub async fn seed_nodes(pool: &PgPool, limit: i32) -> Result<Vec<i32>, sqlx::Error> {
    let rows = sqlx::query_scalar::<_, i32>(
        "SELECT id FROM memory_vectors \
         WHERE importance >= 4.0 \
         ORDER BY access_count DESC \
         LIMIT $1"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Strengthen an edge after traversal (the walk changes the graph)
pub async fn strengthen_edge(pool: &PgPool, edge_id: i32, delta: f32) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE memory_edges SET \
         weight = LEAST(1.0, weight + $1), \
         traversal_count = traversal_count + 1, \
         last_traversed = NOW() \
         WHERE id = $2"
    )
    .bind(delta)
    .bind(edge_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Batch strengthen multiple edges
pub async fn strengthen_edges(pool: &PgPool, edge_ids: &[i32], delta: f32) -> Result<(), sqlx::Error> {
    if edge_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "UPDATE memory_edges SET \
         weight = LEAST(1.0, weight + $1), \
         traversal_count = traversal_count + 1, \
         last_traversed = NOW() \
         WHERE id = ANY($2)"
    )
    .bind(delta)
    .bind(edge_ids)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get detailed graph stats for API
pub async fn detailed_stats(pool: &PgPool) -> Result<serde_json::Value, sqlx::Error> {
    let stats = graph_stats(pool).await?;

    let edge_types: Vec<(String, i64, f64)> = sqlx::query_as(
        "SELECT edge_type, COUNT(*)::bigint, AVG(weight)::float8 \
         FROM memory_edges GROUP BY edge_type ORDER BY COUNT(*) DESC"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let domain_dist: Vec<(String, i64)> = sqlx::query_as(
        "SELECT COALESCE(domain, 'unknown'), COUNT(*)::bigint \
         FROM memory_vectors GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 15"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    Ok(serde_json::json!({
        "nodes": stats.nodes,
        "edges": stats.edges,
        "avg_edges_per_node": stats.edges as f64 / stats.nodes.max(1) as f64,
        "edge_types": edge_types.iter().map(|(t, c, w)| serde_json::json!({
            "type": t, "count": c, "avg_weight": w
        })).collect::<Vec<_>>(),
        "domain_distribution": domain_dist.iter().map(|(d, c)| (d.clone(), *c)).collect::<std::collections::HashMap<_, _>>(),
    }))
}
