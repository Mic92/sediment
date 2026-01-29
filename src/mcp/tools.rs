//! MCP Tool definitions for Sediment
//!
//! Simplified unified API with 4 tools: store, recall, list, forget

use chrono::DateTime;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::access::AccessTracker;
use crate::db::score_with_decay;
use crate::item::{Item, ItemFilters};
use crate::retry::{RetryConfig, with_retry};
use crate::{Database, ListScope, StoreScope};

use super::protocol::{CallToolResult, Tool};
use super::server::ServerContext;

/// Get all available tools (4 total)
pub fn get_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "store".to_string(),
            description: "Store content for later retrieval. Use for preferences, facts, reference material, docs, or any information worth remembering. Long content is automatically chunked for better search.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title (recommended for long content)"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for categorization"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source attribution (e.g., URL, file path, 'conversation')"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Custom JSON metadata"
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "ISO datetime when this should expire (optional)"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global"],
                        "default": "project",
                        "description": "Where to store: 'project' (current project) or 'global' (all projects)"
                    },
                    "replace": {
                        "type": "string",
                        "description": "ID of an existing item to replace (atomically delete before storing)"
                    }
                },
                "required": ["content"]
            }),
        },
        Tool {
            name: "recall".to_string(),
            description: "Search stored content by semantic similarity. Returns matching items with relevant excerpts for chunked content.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (semantic search)"
                    },
                    "limit": {
                        "type": "number",
                        "default": 5,
                        "description": "Maximum number of results"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags (any match)"
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum similarity threshold (0.0-1.0). Lower values return more results."
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "list".to_string(),
            description: "List stored items with optional filtering.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "number",
                        "default": 10,
                        "description": "Maximum number of results"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global", "all"],
                        "default": "project",
                        "description": "Which items to list: 'project', 'global', or 'all'"
                    }
                }
            }),
        },
        Tool {
            name: "forget".to_string(),
            description: "Delete a stored item by its ID.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The item ID to delete"
                    }
                },
                "required": ["id"]
            }),
        },
    ]
}

// ========== Parameter Structs ==========

/// Parameters for store tool
#[derive(Debug, Deserialize)]
pub struct StoreParams {
    pub content: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    /// ID of an existing item to replace (atomically delete before storing)
    #[serde(default)]
    pub replace: Option<String>,
}

/// Parameters for recall tool
#[derive(Debug, Deserialize)]
pub struct RecallParams {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub min_similarity: Option<f32>,
}

/// Parameters for list tool
#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Parameters for forget tool
#[derive(Debug, Deserialize)]
pub struct ForgetParams {
    pub id: String,
}

// ========== Tool Execution ==========

/// Execute a tool with retry logic and fresh DB connection per call.
///
/// Each tool call:
/// 1. Opens a fresh database connection (using the shared embedder)
/// 2. Executes the operation
/// 3. Retries on transient failures with exponential backoff
pub async fn execute_tool(ctx: &ServerContext, name: &str, args: Option<Value>) -> CallToolResult {
    let config = RetryConfig::default();

    // Clone args for potential retries (args is consumed on each attempt)
    let args_for_retry = args.clone();

    let result = with_retry(&config, || {
        let ctx_ref = ctx;
        let name_ref = name;
        let args_clone = args_for_retry.clone();

        async move {
            // Open fresh connection with shared embedder
            let mut db = Database::open_with_embedder(
                &ctx_ref.db_path,
                ctx_ref.project_id.clone(),
                ctx_ref.embedder.clone(),
            )
            .await
            .map_err(|e| format!("Failed to open database: {}", e))?;

            // Open access tracker
            let tracker = AccessTracker::open(&ctx_ref.access_db_path)
                .map_err(|e| format!("Failed to open access tracker: {}", e))?;

            // Execute the tool
            let result = match name_ref {
                "store" => execute_store(&mut db, args_clone).await,
                "recall" => execute_recall(&mut db, &tracker, args_clone).await,
                "list" => execute_list(&mut db, args_clone).await,
                "forget" => execute_forget(&mut db, args_clone).await,
                _ => return Ok(CallToolResult::error(format!("Unknown tool: {}", name_ref))),
            };

            // Check if result indicates an error that should be retried
            if result.is_error.unwrap_or(false) {
                // Check if it's a connection-related error worth retrying
                if let Some(content) = result.content.first() {
                    if is_retryable_error(&content.text) {
                        return Err(content.text.clone());
                    }
                }
            }

            Ok(result)
        }
    })
    .await;

    match result {
        Ok(call_result) => call_result,
        Err(e) => CallToolResult::error(format!("Operation failed after retries: {}", e)),
    }
}

/// Check if an error message indicates a transient/retryable failure
fn is_retryable_error(error_msg: &str) -> bool {
    let retryable_patterns = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "resource busy",
        "lock",
        "I/O error",
        "Failed to open",
        "Failed to connect",
    ];

    let lower = error_msg.to_lowercase();
    retryable_patterns
        .iter()
        .any(|p| lower.contains(&p.to_lowercase()))
}

// ========== Tool Implementations ==========

async fn execute_store(db: &mut Database, args: Option<Value>) -> CallToolResult {
    let params: StoreParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Parse scope
    let scope = params
        .scope
        .as_deref()
        .map(|s| s.parse::<StoreScope>())
        .transpose();

    let scope = match scope {
        Ok(s) => s.unwrap_or(StoreScope::Project),
        Err(e) => return CallToolResult::error(e),
    };

    // Parse expires_at if provided
    let expires_at = if let Some(ref exp_str) = params.expires_at {
        match DateTime::parse_from_rfc3339(exp_str) {
            Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
            Err(e) => return CallToolResult::error(format!("Invalid expires_at: {}", e)),
        }
    } else {
        None
    };

    // Handle replace: delete the existing item first
    if let Some(ref replace_id) = params.replace {
        match db.delete_item(replace_id).await {
            Ok(true) => {
                // Item was deleted, continue with storing new one
            }
            Ok(false) => {
                return CallToolResult::error(format!(
                    "Cannot replace: item not found: {}",
                    replace_id
                ));
            }
            Err(e) => {
                return CallToolResult::error(format!("Failed to delete item for replace: {}", e));
            }
        }
    }

    // Build item
    let mut item = Item::new(&params.content).with_tags(params.tags.unwrap_or_default());

    if let Some(title) = params.title {
        item = item.with_title(title);
    }

    if let Some(source) = params.source {
        item = item.with_source(source);
    }

    if let Some(metadata) = params.metadata {
        item = item.with_metadata(metadata);
    }

    if let Some(exp) = expires_at {
        item = item.with_expires_at(exp);
    }

    // Set project_id based on scope
    if scope == StoreScope::Project {
        if let Some(project_id) = db.project_id() {
            item = item.with_project_id(project_id);
        }
    }

    match db.store_item(item).await {
        Ok(store_result) => {
            let mut result = json!({
                "success": true,
                "id": store_result.id,
                "message": format!("Stored in {} scope", scope)
            });

            // Include potential conflicts if any
            if !store_result.potential_conflicts.is_empty() {
                let conflicts: Vec<Value> = store_result
                    .potential_conflicts
                    .iter()
                    .map(|c| {
                        json!({
                            "id": c.id,
                            "content": c.content,
                            "similarity": format!("{:.2}", c.similarity)
                        })
                    })
                    .collect();
                result["potential_conflicts"] = json!(conflicts);
            }

            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Err(e) => CallToolResult::error(format!("Failed to store: {}", e)),
    }
}

async fn execute_recall(
    db: &mut Database,
    tracker: &AccessTracker,
    args: Option<Value>,
) -> CallToolResult {
    let params: RecallParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    let limit = params.limit.unwrap_or(5);
    let min_similarity = params.min_similarity.unwrap_or(0.3);

    let mut filters = ItemFilters::new().with_min_similarity(min_similarity);

    if let Some(tags) = params.tags {
        filters = filters.with_tags(tags);
    }

    match db.search_items(&params.query, limit, filters).await {
        Ok(mut results) => {
            if results.is_empty() {
                return CallToolResult::success("No items found matching your query.");
            }

            // Fetch access records for decay scoring
            let item_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            let access_records = tracker.get_accesses(&item_ids).unwrap_or_default();

            let now = chrono::Utc::now().timestamp();

            // Re-score with decay
            for result in &mut results {
                let created_at = result.created_at.timestamp();
                let (access_count, last_accessed) = match access_records.get(&result.id) {
                    Some(rec) => (rec.access_count, Some(rec.last_accessed_at)),
                    None => (0, None),
                };

                result.similarity = score_with_decay(
                    result.similarity,
                    now,
                    created_at,
                    access_count,
                    last_accessed,
                );
            }

            // Re-sort by decay-adjusted score
            results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

            // Record access for returned items
            for result in &results {
                let created_at = result.created_at.timestamp();
                let _ = tracker.record_access(&result.id, created_at);
            }

            let formatted: Vec<Value> = results
                .iter()
                .map(|r| {
                    let mut obj = json!({
                        "id": r.id,
                        "content": r.content,
                        "similarity": format!("{:.2}", r.similarity),
                        "created": r.created_at.to_rfc3339(),
                    });

                    // Add optional fields
                    if let Some(ref excerpt) = r.relevant_excerpt {
                        obj["relevant_excerpt"] = json!(excerpt);
                    }
                    if !r.tags.is_empty() {
                        obj["tags"] = json!(r.tags);
                    }
                    if let Some(ref source) = r.source {
                        obj["source"] = json!(source);
                    }

                    obj
                })
                .collect();

            let result = json!({
                "count": results.len(),
                "results": formatted
            });

            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Err(e) => CallToolResult::error(format!("Search failed: {}", e)),
    }
}

async fn execute_list(db: &mut Database, args: Option<Value>) -> CallToolResult {
    let params: ListParams =
        args.and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or(ListParams {
                tags: None,
                limit: None,
                scope: None,
            });

    let limit = params.limit.unwrap_or(10);

    let mut filters = ItemFilters::new();

    if let Some(tags) = params.tags {
        filters = filters.with_tags(tags);
    }

    // Parse scope
    let scope = params
        .scope
        .as_deref()
        .map(|s| s.parse::<ListScope>())
        .transpose();

    let scope = match scope {
        Ok(s) => s.unwrap_or(ListScope::Project),
        Err(e) => return CallToolResult::error(e),
    };

    match db.list_items(filters, Some(limit), scope).await {
        Ok(items) => {
            if items.is_empty() {
                CallToolResult::success("No items stored yet.")
            } else {
                let formatted: Vec<Value> = items
                    .iter()
                    .map(|item| {
                        let content_preview = truncate(&item.content, 100);
                        let mut obj = json!({
                            "id": item.id,
                            "content": content_preview,
                            "created": item.created_at.to_rfc3339(),
                        });

                        if let Some(ref title) = item.title {
                            obj["title"] = json!(title);
                        }
                        if !item.tags.is_empty() {
                            obj["tags"] = json!(item.tags);
                        }
                        if item.is_chunked {
                            obj["chunked"] = json!(true);
                        }

                        obj
                    })
                    .collect();

                let result = json!({
                    "count": items.len(),
                    "items": formatted
                });

                CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
            }
        }
        Err(e) => CallToolResult::error(format!("Failed to list items: {}", e)),
    }
}

async fn execute_forget(db: &mut Database, args: Option<Value>) -> CallToolResult {
    let params: ForgetParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    match db.delete_item(&params.id).await {
        Ok(true) => {
            let result = json!({
                "success": true,
                "message": format!("Deleted item: {}", params.id)
            });
            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Ok(false) => CallToolResult::error(format!("Item not found: {}", params.id)),
        Err(e) => CallToolResult::error(format!("Failed to delete: {}", e)),
    }
}

// ========== Utilities ==========

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
