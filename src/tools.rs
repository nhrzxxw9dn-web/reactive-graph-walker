//! Tool calling framework — RGW can use tools to interact with the world.
//!
//! Tools:
//!   - web_search: search the internet, results become graph nodes
//!   - web_fetch: scrape a URL, content becomes a graph node
//!   - memory_store: create a new memory node in the graph
//!   - edge_create: connect two nodes
//!   - code_exec: execute code in a sandbox (future)
//!   - speech_say: speak text via TTS (future)
//!
//! Tools are invoked by the walker when it detects gaps or needs
//! external information. Results flow back through the self-model.

use serde::{Deserialize, Serialize};

/// A tool that RGW can invoke
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Result of a tool invocation
#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub content: String,
    pub metadata: serde_json::Value,
}

/// Available tools
pub fn available_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "code_exec".into(),
            description: "Execute Python code in a sandbox. Returns stdout/stderr.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout_secs": {"type": "integer", "description": "Max execution time (default 10)"}
                },
                "required": ["code"]
            }),
        },
        Tool {
            name: "web_search".into(),
            description: "Search the internet. Results become nodes in the graph.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "web_fetch".into(),
            description: "Fetch and read a web page. Content becomes a graph node.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }),
        },
        Tool {
            name: "memory_store".into(),
            description: "Store a new memory/concept in the graph.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "domain": {"type": "string"},
                    "importance": {"type": "number"}
                },
                "required": ["content"]
            }),
        },
        Tool {
            name: "edge_create".into(),
            description: "Create a connection between two nodes.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "target_id": {"type": "integer"},
                    "edge_type": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["source_id", "target_id", "edge_type"]
            }),
        },
    ]
}

/// Execute a tool by name
pub async fn execute_tool(
    name: &str,
    params: serde_json::Value,
    pool: &sqlx::PgPool,
) -> ToolResult {
    match name {
        "code_exec" => code_exec(params).await,
        "web_search" => web_search(params).await,
        "web_fetch" => web_fetch(params).await,
        "memory_store" => memory_store(params, pool).await,
        "edge_create" => edge_create(params, pool).await,
        _ => ToolResult {
            tool: name.into(),
            success: false,
            content: format!("Unknown tool: {}", name),
            metadata: serde_json::json!({}),
        },
    }
}

/// Execute Python code in a sandbox (subprocess with timeout)
async fn code_exec(params: serde_json::Value) -> ToolResult {
    let code = params["code"].as_str().unwrap_or("");
    let timeout = params["timeout_secs"].as_u64().unwrap_or(10);

    if code.is_empty() {
        return ToolResult {
            tool: "code_exec".into(),
            success: false,
            content: "Empty code".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Security: reject dangerous patterns
    let forbidden = ["os.system", "subprocess", "shutil.rmtree", "__import__", "eval(", "exec(", "open("];
    for pattern in &forbidden {
        if code.contains(pattern) {
            return ToolResult {
                tool: "code_exec".into(),
                success: false,
                content: format!("Forbidden: code contains '{}'", pattern),
                metadata: serde_json::json!({"blocked": true}),
            };
        }
    }

    use tokio::process::Command;

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout),
        Command::new("python3")
            .arg("-c")
            .arg(code)
            .output()
    ).await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let success = output.status.success();

            let content = if success {
                if stdout.is_empty() { "(no output)".to_string() } else { stdout.clone() }
            } else {
                format!("Error:\n{}", stderr)
            };

            ToolResult {
                tool: "code_exec".into(),
                success,
                content: content[..content.len().min(2000)].to_string(),
                metadata: serde_json::json!({
                    "exit_code": output.status.code(),
                    "stdout_len": stdout.len(),
                    "stderr_len": stderr.len(),
                }),
            }
        }
        Ok(Err(e)) => ToolResult {
            tool: "code_exec".into(),
            success: false,
            content: format!("Failed to spawn: {}", e),
            metadata: serde_json::json!({}),
        },
        Err(_) => ToolResult {
            tool: "code_exec".into(),
            success: false,
            content: format!("Timeout after {}s", timeout),
            metadata: serde_json::json!({"timeout": true}),
        },
    }
}

/// Search the web using a search API
async fn web_search(params: serde_json::Value) -> ToolResult {
    let query = params["query"].as_str().unwrap_or("");
    if query.is_empty() {
        return ToolResult {
            tool: "web_search".into(),
            success: false,
            content: "Empty query".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Use DuckDuckGo instant answer API (free, no key needed)
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding::encode(query)
    );

    match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                let abstract_text = data["AbstractText"].as_str().unwrap_or("");
                let heading = data["Heading"].as_str().unwrap_or("");
                let source = data["AbstractSource"].as_str().unwrap_or("");

                let content = if !abstract_text.is_empty() {
                    format!("{}: {}", heading, abstract_text)
                } else {
                    // Try related topics
                    let topics: Vec<String> = data["RelatedTopics"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .take(5)
                        .filter_map(|t| t["Text"].as_str().map(|s| s.to_string()))
                        .collect();
                    if topics.is_empty() {
                        format!("No results for: {}", query)
                    } else {
                        topics.join("\n")
                    }
                };

                ToolResult {
                    tool: "web_search".into(),
                    success: !content.contains("No results"),
                    content,
                    metadata: serde_json::json!({
                        "query": query,
                        "source": source,
                        "heading": heading,
                    }),
                }
            } else {
                ToolResult {
                    tool: "web_search".into(),
                    success: false,
                    content: "Failed to parse search results".into(),
                    metadata: serde_json::json!({"query": query}),
                }
            }
        }
        Err(e) => ToolResult {
            tool: "web_search".into(),
            success: false,
            content: format!("Search failed: {}", e),
            metadata: serde_json::json!({"query": query}),
        },
    }
}

/// Fetch a web page
async fn web_fetch(params: serde_json::Value) -> ToolResult {
    let url = params["url"].as_str().unwrap_or("");
    if url.is_empty() {
        return ToolResult {
            tool: "web_fetch".into(),
            success: false,
            content: "Empty URL".into(),
            metadata: serde_json::json!({}),
        };
    }

    match reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => {
                    // Truncate to first 2000 chars (graph nodes shouldn't be huge)
                    let content = if body.len() > 2000 {
                        format!("{}...", &body[..2000])
                    } else {
                        body
                    };
                    ToolResult {
                        tool: "web_fetch".into(),
                        success: status < 400,
                        content,
                        metadata: serde_json::json!({"url": url, "status": status}),
                    }
                }
                Err(e) => ToolResult {
                    tool: "web_fetch".into(),
                    success: false,
                    content: format!("Failed to read body: {}", e),
                    metadata: serde_json::json!({"url": url}),
                },
            }
        }
        Err(e) => ToolResult {
            tool: "web_fetch".into(),
            success: false,
            content: format!("Fetch failed: {}", e),
            metadata: serde_json::json!({"url": url}),
        },
    }
}

/// Store a new memory node
async fn memory_store(params: serde_json::Value, pool: &sqlx::PgPool) -> ToolResult {
    let content = params["content"].as_str().unwrap_or("");
    let domain = params["domain"].as_str().unwrap_or("unknown");
    let importance = params["importance"].as_f64().unwrap_or(5.0) as f32;

    if content.is_empty() {
        return ToolResult {
            tool: "memory_store".into(),
            success: false,
            content: "Empty content".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Store via direct SQL (nodes are memory_vectors rows)
    match sqlx::query(
        "INSERT INTO memory_vectors (doc_id, document, domain, importance, source_type, stored_at, created_at, updated_at) \
         VALUES ($1, $2, $3, $4, 'rgw_tool', extract(epoch from now()), NOW(), NOW()) \
         RETURNING id"
    )
    .bind(format!("rgw_{}", chrono::Utc::now().timestamp()))
    .bind(content)
    .bind(domain)
    .bind(importance)
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let id: i32 = row.get("id");
            ToolResult {
                tool: "memory_store".into(),
                success: true,
                content: format!("Stored as node {} (domain: {})", id, domain),
                metadata: serde_json::json!({"node_id": id, "domain": domain}),
            }
        }
        Err(e) => ToolResult {
            tool: "memory_store".into(),
            success: false,
            content: format!("Store failed: {}", e),
            metadata: serde_json::json!({}),
        },
    }
}

/// Create an edge between nodes
async fn edge_create(params: serde_json::Value, pool: &sqlx::PgPool) -> ToolResult {
    let source = params["source_id"].as_i64().unwrap_or(0) as i32;
    let target = params["target_id"].as_i64().unwrap_or(0) as i32;
    let edge_type = params["edge_type"].as_str().unwrap_or("related");
    let weight = params["weight"].as_f64().unwrap_or(0.5) as f32;

    match crate::db::create_edge(pool, source, target, edge_type, weight, 0.0).await {
        Ok(edge_id) => ToolResult {
            tool: "edge_create".into(),
            success: true,
            content: format!("Edge {} → {} (type: {}, id: {})", source, target, edge_type, edge_id),
            metadata: serde_json::json!({"edge_id": edge_id}),
        },
        Err(e) => ToolResult {
            tool: "edge_create".into(),
            success: false,
            content: format!("Edge creation failed: {}", e),
            metadata: serde_json::json!({}),
        },
    }
}

// Need urlencoding for search queries
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

use sqlx::Row;
use chrono;
