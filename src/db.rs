//! Database module using LanceDB for vector storage
//!
//! Provides a simple interface for storing and searching items
//! using LanceDB's native vector search capabilities.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use tracing::{debug, info};

use crate::chunker::{ChunkingConfig, chunk_content};
use crate::document::ContentType;
use crate::embedder::{EMBEDDING_DIM, Embedder};
use crate::error::{Result, SedimentError};
use crate::item::{Chunk, ConflictInfo, Item, ItemFilters, SearchResult, StoreResult};

/// Threshold for auto-chunking (in characters)
const CHUNK_THRESHOLD: usize = 1000;

/// Similarity threshold for conflict detection
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Maximum number of conflicts to return
const CONFLICT_SEARCH_LIMIT: usize = 5;

/// Database wrapper for LanceDB
pub struct Database {
    db: lancedb::Connection,
    embedder: Arc<Embedder>,
    project_id: Option<String>,
    items_table: Option<Table>,
    chunks_table: Option<Table>,
}

/// Database statistics
#[derive(Debug, Default, Clone)]
pub struct DatabaseStats {
    pub item_count: usize,
    pub chunk_count: usize,
}

// Arrow schema builders
fn item_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, true),
        Field::new("tags", DataType::Utf8, true), // JSON array as string
        Field::new("source", DataType::Utf8, true),
        Field::new("metadata", DataType::Utf8, true), // JSON as string
        Field::new("project_id", DataType::Utf8, true),
        Field::new("is_chunked", DataType::Boolean, false),
        Field::new("expires_at", DataType::Int64, true), // Unix timestamp
        Field::new("created_at", DataType::Int64, false), // Unix timestamp
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ])
}

fn chunk_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("item_id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("context", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ])
}

impl Database {
    /// Open or create a database at the given path
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_project(path, None).await
    }

    /// Open or create a database at the given path with a project ID
    pub async fn open_with_project(
        path: impl Into<PathBuf>,
        project_id: Option<String>,
    ) -> Result<Self> {
        let embedder = Arc::new(Embedder::new()?);
        Self::open_with_embedder(path, project_id, embedder).await
    }

    /// Open or create a database with a pre-existing embedder.
    ///
    /// This constructor is useful for connection pooling scenarios where
    /// the expensive embedder should be loaded once and shared across
    /// multiple database connections.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    /// * `project_id` - Optional project ID for scoped operations
    /// * `embedder` - Shared embedder instance
    pub async fn open_with_embedder(
        path: impl Into<PathBuf>,
        project_id: Option<String>,
        embedder: Arc<Embedder>,
    ) -> Result<Self> {
        let path = path.into();
        info!("Opening database at {:?}", path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SedimentError::Database(format!("Failed to create database directory: {}", e))
            })?;
        }

        let db = connect(path.to_str().unwrap())
            .execute()
            .await
            .map_err(|e| {
                SedimentError::Database(format!("Failed to connect to database: {}", e))
            })?;

        let mut database = Self {
            db,
            embedder,
            project_id,
            items_table: None,
            chunks_table: None,
        };

        database.ensure_tables().await?;

        Ok(database)
    }

    /// Set the current project ID for scoped operations
    pub fn set_project_id(&mut self, project_id: Option<String>) {
        self.project_id = project_id;
    }

    /// Get the current project ID
    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    /// Ensure all required tables exist
    async fn ensure_tables(&mut self) -> Result<()> {
        // Check for existing tables
        let table_names = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to list tables: {}", e)))?;

        // Items table
        if table_names.contains(&"items".to_string()) {
            self.items_table =
                Some(self.db.open_table("items").execute().await.map_err(|e| {
                    SedimentError::Database(format!("Failed to open items: {}", e))
                })?);
        }

        // Chunks table
        if table_names.contains(&"chunks".to_string()) {
            self.chunks_table =
                Some(self.db.open_table("chunks").execute().await.map_err(|e| {
                    SedimentError::Database(format!("Failed to open chunks: {}", e))
                })?);
        }

        Ok(())
    }

    /// Get or create the items table
    async fn get_items_table(&mut self) -> Result<&Table> {
        if self.items_table.is_none() {
            let schema = Arc::new(item_schema());
            let table = self
                .db
                .create_empty_table("items", schema)
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to create items table: {}", e))
                })?;
            self.items_table = Some(table);
        }
        Ok(self.items_table.as_ref().unwrap())
    }

    /// Get or create the chunks table
    async fn get_chunks_table(&mut self) -> Result<&Table> {
        if self.chunks_table.is_none() {
            let schema = Arc::new(chunk_schema());
            let table = self
                .db
                .create_empty_table("chunks", schema)
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to create chunks table: {}", e))
                })?;
            self.chunks_table = Some(table);
        }
        Ok(self.chunks_table.as_ref().unwrap())
    }

    // ==================== Item Operations ====================

    /// Store an item with automatic chunking for long content
    ///
    /// Returns a `StoreResult` containing the new item ID and any potential conflicts
    /// (items with similarity >= 0.85 to the new content).
    pub async fn store_item(&mut self, mut item: Item) -> Result<StoreResult> {
        // Set project_id if not already set and we have a current project
        if item.project_id.is_none() {
            item.project_id = self.project_id.clone();
        }

        // Check for potential conflicts before storing
        let potential_conflicts = self
            .find_similar_items(
                &item.content,
                CONFLICT_SIMILARITY_THRESHOLD,
                CONFLICT_SEARCH_LIMIT,
            )
            .await
            .unwrap_or_default();

        // Determine if we need to chunk
        let should_chunk = item.content.len() > CHUNK_THRESHOLD;
        item.is_chunked = should_chunk;

        // Generate item embedding
        let embedding_text = item.embedding_text();
        let embedding = self.embedder.embed(&embedding_text)?;
        item.embedding = embedding;

        // Store the item
        let table = self.get_items_table().await?;
        let batch = item_to_batch(&item)?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], Arc::new(item_schema()));

        table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to store item: {}", e)))?;

        // If chunking is needed, create and store chunks
        if should_chunk {
            let embedder = self.embedder.clone();
            let chunks_table = self.get_chunks_table().await?;

            // Detect content type for smart chunking
            let content_type = detect_content_type(&item.content);
            let config = ChunkingConfig::default();
            let chunk_results = chunk_content(&item.content, content_type, &config);

            for (i, chunk_result) in chunk_results.iter().enumerate() {
                let mut chunk = Chunk::new(&item.id, i, &chunk_result.content);

                if let Some(ctx) = &chunk_result.context {
                    chunk = chunk.with_context(ctx);
                }

                let chunk_embedding = embedder.embed(&chunk.content)?;
                chunk.embedding = chunk_embedding;

                let chunk_batch = chunk_to_batch(&chunk)?;
                let batches =
                    RecordBatchIterator::new(vec![Ok(chunk_batch)], Arc::new(chunk_schema()));

                chunks_table
                    .add(Box::new(batches))
                    .execute()
                    .await
                    .map_err(|e| {
                        SedimentError::Database(format!("Failed to store chunk: {}", e))
                    })?;
            }

            debug!(
                "Stored item: {} with {} chunks",
                item.id,
                chunk_results.len()
            );
        } else {
            debug!("Stored item: {} (no chunking)", item.id);
        }

        Ok(StoreResult {
            id: item.id,
            potential_conflicts,
        })
    }

    /// Search items by semantic similarity
    pub async fn search_items(
        &mut self,
        query: &str,
        limit: usize,
        filters: ItemFilters,
    ) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embedder.embed(query)?;
        let min_similarity = filters.min_similarity.unwrap_or(0.3);

        // We need to search both items and chunks, then merge results
        let mut results_map: std::collections::HashMap<String, (SearchResult, f32)> =
            std::collections::HashMap::new();

        // Search items table directly (for non-chunked items and chunked items by title)
        if let Some(table) = &self.items_table {
            let mut filter_parts = Vec::new();

            if !filters.include_expired {
                let now = Utc::now().timestamp();
                filter_parts.push(format!("(expires_at IS NULL OR expires_at > {})", now));
            }

            let mut query_builder = table
                .vector_search(query_embedding.clone())
                .map_err(|e| SedimentError::Database(format!("Failed to build search: {}", e)))?
                .limit(limit * 2);

            if !filter_parts.is_empty() {
                let filter_str = filter_parts.join(" AND ");
                query_builder = query_builder.only_if(filter_str);
            }

            let results = query_builder
                .execute()
                .await
                .map_err(|e| SedimentError::Database(format!("Search failed: {}", e)))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to collect results: {}", e))
                })?;

            for batch in results {
                let items = batch_to_items(&batch)?;
                let distances = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

                for (i, item) in items.into_iter().enumerate() {
                    let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                    let similarity = 1.0 / (1.0 + distance);

                    if similarity < min_similarity {
                        continue;
                    }

                    // Apply tag filter
                    if let Some(ref filter_tags) = filters.tags {
                        if !filter_tags.iter().any(|t| item.tags.contains(t)) {
                            continue;
                        }
                    }

                    // Apply project boosting
                    let boosted_similarity = boost_similarity(
                        similarity,
                        item.project_id.as_deref(),
                        self.project_id.as_deref(),
                    );

                    let result = SearchResult::from_item(&item, boosted_similarity);
                    results_map
                        .entry(item.id.clone())
                        .or_insert((result, boosted_similarity));
                }
            }
        }

        // Search chunks table (for chunked items)
        if let Some(chunks_table) = &self.chunks_table {
            let chunk_results = chunks_table
                .vector_search(query_embedding)
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to build chunk search: {}", e))
                })?
                .limit(limit * 3)
                .execute()
                .await
                .map_err(|e| SedimentError::Database(format!("Chunk search failed: {}", e)))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to collect chunk results: {}", e))
                })?;

            // Group chunks by item and find best chunk for each item
            let mut chunk_matches: std::collections::HashMap<String, (String, f32)> =
                std::collections::HashMap::new();

            for batch in chunk_results {
                let chunks = batch_to_chunks(&batch)?;
                let distances = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

                for (i, chunk) in chunks.into_iter().enumerate() {
                    let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                    let similarity = 1.0 / (1.0 + distance);

                    if similarity < min_similarity {
                        continue;
                    }

                    // Keep track of best matching chunk per item
                    chunk_matches
                        .entry(chunk.item_id.clone())
                        .and_modify(|(content, best_sim)| {
                            if similarity > *best_sim {
                                *content = chunk.content.clone();
                                *best_sim = similarity;
                            }
                        })
                        .or_insert((chunk.content.clone(), similarity));
                }
            }

            // Fetch parent items for chunk matches
            for (item_id, (excerpt, chunk_similarity)) in chunk_matches {
                if let Some(item) = self.get_item(&item_id).await? {
                    // Apply tag filter
                    if let Some(ref filter_tags) = filters.tags {
                        if !filter_tags.iter().any(|t| item.tags.contains(t)) {
                            continue;
                        }
                    }

                    // Apply project boosting
                    let boosted_similarity = boost_similarity(
                        chunk_similarity,
                        item.project_id.as_deref(),
                        self.project_id.as_deref(),
                    );

                    let result =
                        SearchResult::from_item_with_excerpt(&item, boosted_similarity, excerpt);

                    // Update if this chunk-based result is better
                    results_map
                        .entry(item_id)
                        .and_modify(|(existing, existing_sim)| {
                            if boosted_similarity > *existing_sim {
                                *existing = result.clone();
                                *existing_sim = boosted_similarity;
                            }
                        })
                        .or_insert((result, boosted_similarity));
                }
            }
        }

        // Convert map to sorted vec
        let mut search_results: Vec<SearchResult> =
            results_map.into_values().map(|(r, _)| r).collect();
        search_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        search_results.truncate(limit);

        Ok(search_results)
    }

    /// Find items similar to the given content (for conflict detection)
    ///
    /// This searches the items table directly by content embedding to find
    /// potentially conflicting items before storing new content.
    pub async fn find_similar_items(
        &mut self,
        content: &str,
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<ConflictInfo>> {
        // Generate embedding for the content
        let embedding = self.embedder.embed(content)?;

        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        // Build filter for non-expired items
        let now = Utc::now().timestamp();
        let filter = format!("(expires_at IS NULL OR expires_at > {})", now);

        let results = table
            .vector_search(embedding)
            .map_err(|e| SedimentError::Database(format!("Failed to build search: {}", e)))?
            .limit(limit)
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Search failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect results: {}", e)))?;

        let mut conflicts = Vec::new();

        for batch in results {
            let items = batch_to_items(&batch)?;
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for (i, item) in items.into_iter().enumerate() {
                let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                let similarity = 1.0 / (1.0 + distance);

                if similarity >= min_similarity {
                    conflicts.push(ConflictInfo {
                        id: item.id,
                        content: item.content,
                        similarity,
                    });
                }
            }
        }

        // Sort by similarity descending
        conflicts.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        Ok(conflicts)
    }

    /// List items with optional filters
    pub async fn list_items(
        &mut self,
        filters: ItemFilters,
        limit: Option<usize>,
        scope: crate::ListScope,
    ) -> Result<Vec<Item>> {
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut filter_parts = Vec::new();

        if !filters.include_expired {
            let now = Utc::now().timestamp();
            filter_parts.push(format!("(expires_at IS NULL OR expires_at > {})", now));
        }

        // Apply scope filter
        match scope {
            crate::ListScope::Project => {
                if let Some(ref pid) = self.project_id {
                    filter_parts.push(format!("project_id = '{}'", pid));
                }
            }
            crate::ListScope::Global => {
                filter_parts.push("project_id IS NULL".to_string());
            }
            crate::ListScope::All => {
                // No additional filter
            }
        }

        let mut query = table.query();

        if !filter_parts.is_empty() {
            let filter_str = filter_parts.join(" AND ");
            query = query.only_if(filter_str);
        }

        if let Some(l) = limit {
            query = query.limit(l);
        }

        let results = query
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect: {}", e)))?;

        let mut items = Vec::new();
        for batch in results {
            items.extend(batch_to_items(&batch)?);
        }

        // Apply tag filter
        if let Some(ref filter_tags) = filters.tags {
            items.retain(|item| filter_tags.iter().any(|t| item.tags.contains(t)));
        }

        Ok(items)
    }

    /// Get an item by ID
    pub async fn get_item(&self, id: &str) -> Result<Option<Item>> {
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(None),
        };

        let results = table
            .query()
            .only_if(format!("id = '{}'", id))
            .limit(1)
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect: {}", e)))?;

        for batch in results {
            let items = batch_to_items(&batch)?;
            if let Some(item) = items.into_iter().next() {
                return Ok(Some(item));
            }
        }

        Ok(None)
    }

    /// Delete an item and its chunks
    pub async fn delete_item(&self, id: &str) -> Result<bool> {
        // Delete chunks first
        if let Some(chunks_table) = &self.chunks_table {
            chunks_table
                .delete(&format!("item_id = '{}'", id))
                .await
                .map_err(|e| SedimentError::Database(format!("Delete chunks failed: {}", e)))?;
        }

        // Delete item
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(false),
        };

        table
            .delete(&format!("id = '{}'", id))
            .await
            .map_err(|e| SedimentError::Database(format!("Delete failed: {}", e)))?;

        Ok(true)
    }

    /// Get database statistics
    pub async fn stats(&self) -> Result<DatabaseStats> {
        let mut stats = DatabaseStats::default();

        if let Some(table) = &self.items_table {
            stats.item_count = table
                .count_rows(None)
                .await
                .map_err(|e| SedimentError::Database(format!("Count failed: {}", e)))?;
        }

        if let Some(table) = &self.chunks_table {
            stats.chunk_count = table
                .count_rows(None)
                .await
                .map_err(|e| SedimentError::Database(format!("Count failed: {}", e)))?;
        }

        Ok(stats)
    }
}

// ==================== Helper Functions ====================

/// Apply similarity boosting based on project context.
fn boost_similarity(base: f32, item_project: Option<&str>, current_project: Option<&str>) -> f32 {
    match (item_project, current_project) {
        (Some(m), Some(c)) if m == c => (base * 1.15).min(1.0), // Same project: boost
        (Some(_), Some(_)) => base * 0.95,                      // Different project: slight penalty
        _ => base,                                              // Global or no context
    }
}

/// Detect content type for smart chunking
fn detect_content_type(content: &str) -> ContentType {
    let trimmed = content.trim();

    // Check for JSON
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return ContentType::Json;
        }
    }

    // Check for YAML (common patterns)
    if trimmed.contains(":\n") || trimmed.starts_with("---") {
        // Simple heuristic: looks like YAML if it has key: value patterns
        let lines: Vec<&str> = trimmed.lines().take(5).collect();
        let yaml_like = lines.iter().any(|line| {
            let l = line.trim();
            !l.is_empty() && !l.starts_with('#') && l.contains(':') && !l.starts_with("http")
        });
        if yaml_like {
            return ContentType::Yaml;
        }
    }

    // Check for Markdown (has headers)
    if trimmed.contains("\n# ") || trimmed.starts_with("# ") || trimmed.contains("\n## ") {
        return ContentType::Markdown;
    }

    // Check for code (common patterns)
    let code_patterns = [
        "fn ",
        "pub fn ",
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "var ",
        "import ",
        "export ",
        "struct ",
        "impl ",
        "trait ",
    ];
    if code_patterns.iter().any(|p| trimmed.contains(p)) {
        return ContentType::Code;
    }

    ContentType::Text
}

// ==================== Arrow Conversion Helpers ====================

fn item_to_batch(item: &Item) -> Result<RecordBatch> {
    let schema = Arc::new(item_schema());

    let id = StringArray::from(vec![item.id.as_str()]);
    let content = StringArray::from(vec![item.content.as_str()]);
    let title = StringArray::from(vec![item.title.as_deref()]);
    let tags = StringArray::from(vec![serde_json::to_string(&item.tags).ok()]);
    let source = StringArray::from(vec![item.source.as_deref()]);
    let metadata = StringArray::from(vec![item.metadata.as_ref().map(|m| m.to_string())]);
    let project_id = StringArray::from(vec![item.project_id.as_deref()]);
    let is_chunked = BooleanArray::from(vec![item.is_chunked]);
    let expires_at = Int64Array::from(vec![item.expires_at.map(|t| t.timestamp())]);
    let created_at = Int64Array::from(vec![item.created_at.timestamp()]);

    let vector = create_embedding_array(&item.embedding)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(content),
            Arc::new(title),
            Arc::new(tags),
            Arc::new(source),
            Arc::new(metadata),
            Arc::new(project_id),
            Arc::new(is_chunked),
            Arc::new(expires_at),
            Arc::new(created_at),
            Arc::new(vector),
        ],
    )
    .map_err(|e| SedimentError::Database(format!("Failed to create batch: {}", e)))
}

fn batch_to_items(batch: &RecordBatch) -> Result<Vec<Item>> {
    let mut items = Vec::new();

    let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing id column".to_string()))?;

    let content_col = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing content column".to_string()))?;

    let title_col = batch
        .column_by_name("title")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let tags_col = batch
        .column_by_name("tags")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let source_col = batch
        .column_by_name("source")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let metadata_col = batch
        .column_by_name("metadata")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let project_id_col = batch
        .column_by_name("project_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let is_chunked_col = batch
        .column_by_name("is_chunked")
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

    let expires_at_col = batch
        .column_by_name("expires_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    let created_at_col = batch
        .column_by_name("created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    for i in 0..batch.num_rows() {
        let id = id_col.value(i).to_string();
        let content = content_col.value(i).to_string();

        let title = title_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let tags: Vec<String> = tags_col
            .and_then(|c| {
                if c.is_null(i) {
                    None
                } else {
                    serde_json::from_str(c.value(i)).ok()
                }
            })
            .unwrap_or_default();

        let source = source_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let metadata = metadata_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                serde_json::from_str(c.value(i)).ok()
            }
        });

        let project_id = project_id_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let is_chunked = is_chunked_col.map(|c| c.value(i)).unwrap_or(false);

        let expires_at = expires_at_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(Utc.timestamp_opt(c.value(i), 0).unwrap())
            }
        });

        let created_at = created_at_col
            .map(|c| Utc.timestamp_opt(c.value(i), 0).unwrap())
            .unwrap_or_else(Utc::now);

        let item = Item {
            id,
            content,
            embedding: Vec::new(),
            title,
            tags,
            source,
            metadata,
            project_id,
            is_chunked,
            expires_at,
            created_at,
        };

        items.push(item);
    }

    Ok(items)
}

fn chunk_to_batch(chunk: &Chunk) -> Result<RecordBatch> {
    let schema = Arc::new(chunk_schema());

    let id = StringArray::from(vec![chunk.id.as_str()]);
    let item_id = StringArray::from(vec![chunk.item_id.as_str()]);
    let chunk_index = Int32Array::from(vec![chunk.chunk_index as i32]);
    let content = StringArray::from(vec![chunk.content.as_str()]);
    let context = StringArray::from(vec![chunk.context.as_deref()]);

    let vector = create_embedding_array(&chunk.embedding)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(item_id),
            Arc::new(chunk_index),
            Arc::new(content),
            Arc::new(context),
            Arc::new(vector),
        ],
    )
    .map_err(|e| SedimentError::Database(format!("Failed to create batch: {}", e)))
}

fn batch_to_chunks(batch: &RecordBatch) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();

    let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing id column".to_string()))?;

    let item_id_col = batch
        .column_by_name("item_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing item_id column".to_string()))?;

    let chunk_index_col = batch
        .column_by_name("chunk_index")
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| SedimentError::Database("Missing chunk_index column".to_string()))?;

    let content_col = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing content column".to_string()))?;

    let context_col = batch
        .column_by_name("context")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    for i in 0..batch.num_rows() {
        let id = id_col.value(i).to_string();
        let item_id = item_id_col.value(i).to_string();
        let chunk_index = chunk_index_col.value(i) as usize;
        let content = content_col.value(i).to_string();
        let context = context_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let chunk = Chunk {
            id,
            item_id,
            chunk_index,
            content,
            embedding: Vec::new(),
            context,
        };

        chunks.push(chunk);
    }

    Ok(chunks)
}

fn create_embedding_array(embedding: &[f32]) -> Result<FixedSizeListArray> {
    let values = Float32Array::from(embedding.to_vec());
    let field = Arc::new(Field::new("item", DataType::Float32, true));

    FixedSizeListArray::try_new(field, EMBEDDING_DIM as i32, Arc::new(values), None)
        .map_err(|e| SedimentError::Database(format!("Failed to create vector: {}", e)))
}
