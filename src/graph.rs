//! Graph store using Kuzu for relationship tracking between memories.
//!
//! Provides a property graph layer alongside LanceDB for tracking
//! relationships like RELATED, SUPERSEDES, and CO_ACCESSED between items.

use std::path::Path;

use kuzu::{Connection, Database, SystemConfig, Value};
use tracing::debug;

use crate::error::{Result, SedimentError};

/// A relationship between two memory items in the graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub target_id: String,
    pub rel_type: String,
    pub strength: f64,
}

/// A co-access relationship between two memory items.
#[derive(Debug, Clone)]
pub struct CoAccessEdge {
    pub target_id: String,
    pub count: i64,
}

/// Full connection info for a memory item.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub target_id: String,
    pub rel_type: String,
    pub strength: f64,
    pub count: Option<i64>,
}

/// Kuzu graph store for memory relationships.
pub struct GraphStore {
    db: Database,
}

impl GraphStore {
    /// Open or create a graph database at the given path.
    /// Creates the schema lazily on first access.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|e| {
            SedimentError::Database(format!("Failed to create graph directory: {}", e))
        })?;

        let db = Database::new(
            path.to_str().unwrap_or_default(),
            SystemConfig::default(),
        )
        .map_err(|e| SedimentError::Database(format!("Failed to open graph database: {}", e)))?;

        let store = Self { db };
        store.ensure_schema()?;
        Ok(store)
    }

    /// Create a new connection to the graph database.
    fn conn(&self) -> Result<Connection<'_>> {
        Connection::new(&self.db)
            .map_err(|e| SedimentError::Database(format!("Failed to create graph connection: {}", e)))
    }

    /// Ensure all required node and edge tables exist.
    fn ensure_schema(&self) -> Result<()> {
        let conn = self.conn()?;

        // Create Memory node table if not exists
        let _ = conn.query(
            "CREATE NODE TABLE IF NOT EXISTS Memory (
                id STRING,
                project_id STRING,
                created_at INT64,
                PRIMARY KEY(id)
            )"
        );

        // Create RELATED relationship table if not exists
        let _ = conn.query(
            "CREATE REL TABLE IF NOT EXISTS RELATED (
                FROM Memory TO Memory,
                strength DOUBLE,
                rel_type STRING,
                created_at INT64
            )"
        );

        // Create SUPERSEDES relationship table if not exists
        let _ = conn.query(
            "CREATE REL TABLE IF NOT EXISTS SUPERSEDES (
                FROM Memory TO Memory,
                created_at INT64
            )"
        );

        // Create CO_ACCESSED relationship table if not exists
        let _ = conn.query(
            "CREATE REL TABLE IF NOT EXISTS CO_ACCESSED (
                FROM Memory TO Memory,
                count INT64,
                last_at INT64
            )"
        );

        // Create CLUSTER_SIBLING relationship table if not exists
        let _ = conn.query(
            "CREATE REL TABLE IF NOT EXISTS CLUSTER_SIBLING (
                FROM Memory TO Memory,
                cluster_label STRING,
                created_at INT64
            )"
        );

        Ok(())
    }

    /// Add a Memory node to the graph.
    pub fn add_node(&self, id: &str, project_id: Option<&str>, created_at: i64) -> Result<()> {
        let conn = self.conn()?;
        let pid = project_id.unwrap_or("");

        let mut stmt = conn.prepare(
            "CREATE (m:Memory {id: $id, project_id: $pid, created_at: $ts})"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare add_node: {}", e)))?;

        let result = conn.execute(&mut stmt, vec![
            ("id", Value::String(id.to_string())),
            ("pid", Value::String(pid.to_string())),
            ("ts", Value::Int64(created_at)),
        ]);

        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = format!("{}", e);
                // Ignore duplicate key errors (node already exists)
                if msg.contains("already exists") || msg.contains("duplicate") || msg.contains("PRIMARY KEY") || msg.contains("Violates primary key") {
                    debug!("Node {} already exists in graph", id);
                } else {
                    return Err(SedimentError::Database(format!("Failed to add node: {}", e)));
                }
            }
        }

        debug!("Added graph node: {}", id);
        Ok(())
    }

    /// Ensure a node exists in the graph. Creates it if missing (for backfill).
    pub fn ensure_node_exists(&self, id: &str, project_id: Option<&str>, created_at: i64) -> Result<()> {
        let conn = self.conn()?;

        // Check if node exists
        let mut stmt = conn.prepare(
            "MATCH (m:Memory {id: $id}) RETURN m.id"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare check: {}", e)))?;

        let result = conn.execute(&mut stmt, vec![
            ("id", Value::String(id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to check node: {}", e)))?;

        if result.get_num_tuples() == 0 {
            self.add_node(id, project_id, created_at)?;
        }

        Ok(())
    }

    /// Remove a Memory node and all its edges from the graph.
    pub fn remove_node(&self, id: &str) -> Result<()> {
        let conn = self.conn()?;

        let mut stmt = conn.prepare(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare remove_node: {}", e)))?;

        conn.execute(&mut stmt, vec![
            ("id", Value::String(id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to remove node: {}", e)))?;

        debug!("Removed graph node: {}", id);
        Ok(())
    }

    /// Add a RELATED edge between two Memory nodes.
    pub fn add_related_edge(&self, from_id: &str, to_id: &str, strength: f64, rel_type: &str) -> Result<()> {
        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();

        let mut stmt = conn.prepare(
            "MATCH (a:Memory {id: $from_id}), (b:Memory {id: $to_id})
             CREATE (a)-[:RELATED {strength: $str, rel_type: $rt, created_at: $ts}]->(b)"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare add_edge: {}", e)))?;

        conn.execute(&mut stmt, vec![
            ("from_id", Value::String(from_id.to_string())),
            ("to_id", Value::String(to_id.to_string())),
            ("str", Value::Double(strength)),
            ("rt", Value::String(rel_type.to_string())),
            ("ts", Value::Int64(now)),
        ]).map_err(|e| SedimentError::Database(format!("Failed to add related edge: {}", e)))?;

        debug!("Added RELATED edge: {} -> {} ({})", from_id, to_id, rel_type);
        Ok(())
    }

    /// Add a SUPERSEDES edge from new_id to old_id.
    pub fn add_supersedes_edge(&self, new_id: &str, old_id: &str) -> Result<()> {
        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();

        let mut stmt = conn.prepare(
            "MATCH (a:Memory {id: $new_id}), (b:Memory {id: $old_id})
             CREATE (a)-[:SUPERSEDES {created_at: $ts}]->(b)"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare supersedes edge: {}", e)))?;

        conn.execute(&mut stmt, vec![
            ("new_id", Value::String(new_id.to_string())),
            ("old_id", Value::String(old_id.to_string())),
            ("ts", Value::Int64(now)),
        ]).map_err(|e| SedimentError::Database(format!("Failed to add supersedes edge: {}", e)))?;

        debug!("Added SUPERSEDES edge: {} -> {}", new_id, old_id);
        Ok(())
    }

    /// Get 1-hop neighbors of the given item IDs via RELATED or SUPERSEDES edges.
    /// Returns (neighbor_id, rel_type, strength) tuples.
    pub fn get_neighbors(&self, ids: &[&str], min_strength: f64) -> Result<Vec<(String, String, f64)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn()?;
        let mut results = Vec::new();

        for id in ids {
            // Query RELATED edges
            let mut stmt = conn.prepare(
                "MATCH (a:Memory {id: $id})-[r:RELATED]-(b:Memory)
                 WHERE r.strength >= $min_str
                 RETURN b.id, r.rel_type, r.strength"
            ).map_err(|e| SedimentError::Database(format!("Failed to prepare neighbors query: {}", e)))?;

            let result = conn.execute(&mut stmt, vec![
                ("id", Value::String(id.to_string())),
                ("min_str", Value::Double(min_strength)),
            ]).map_err(|e| SedimentError::Database(format!("Failed to query neighbors: {}", e)))?;

            for row in result {
                if let (Some(bid), Some(rt), Some(str_val)) = (
                    extract_string(&row, 0),
                    extract_string(&row, 1),
                    extract_double(&row, 2),
                ) {
                    results.push((bid, rt, str_val));
                }
            }

            // Query SUPERSEDES edges
            let mut stmt2 = conn.prepare(
                "MATCH (a:Memory {id: $id})-[r:SUPERSEDES]-(b:Memory)
                 RETURN b.id, 'supersedes', 1.0"
            ).map_err(|e| SedimentError::Database(format!("Failed to prepare supersedes query: {}", e)))?;

            let result2 = conn.execute(&mut stmt2, vec![
                ("id", Value::String(id.to_string())),
            ]).map_err(|e| SedimentError::Database(format!("Failed to query supersedes: {}", e)))?;

            for row in result2 {
                if let Some(bid) = extract_string(&row, 0) {
                    results.push((bid, "supersedes".to_string(), 1.0));
                }
            }
        }

        Ok(results)
    }

    /// Record co-access between pairs of item IDs.
    /// Creates or increments CO_ACCESSED edges.
    pub fn record_co_access(&self, item_ids: &[String]) -> Result<()> {
        if item_ids.len() < 2 {
            return Ok(());
        }

        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();

        for i in 0..item_ids.len() {
            for j in (i + 1)..item_ids.len() {
                let a = &item_ids[i];
                let b = &item_ids[j];

                // Try to increment existing edge
                let mut stmt = conn.prepare(
                    "MATCH (a:Memory {id: $a_id})-[r:CO_ACCESSED]->(b:Memory {id: $b_id})
                     SET r.count = r.count + 1, r.last_at = $now
                     RETURN r.count"
                ).map_err(|e| SedimentError::Database(format!("Failed to prepare co-access update: {}", e)))?;

                let result = conn.execute(&mut stmt, vec![
                    ("a_id", Value::String(a.clone())),
                    ("b_id", Value::String(b.clone())),
                    ("now", Value::Int64(now)),
                ]).map_err(|e| SedimentError::Database(format!("Failed to update co-access: {}", e)))?;

                if result.get_num_tuples() == 0 {
                    // No existing edge, create one
                    let mut create_stmt = conn.prepare(
                        "MATCH (a:Memory {id: $a_id}), (b:Memory {id: $b_id})
                         CREATE (a)-[:CO_ACCESSED {count: 1, last_at: $now}]->(b)"
                    ).map_err(|e| SedimentError::Database(format!("Failed to prepare co-access create: {}", e)))?;

                    let _ = conn.execute(&mut create_stmt, vec![
                        ("a_id", Value::String(a.clone())),
                        ("b_id", Value::String(b.clone())),
                        ("now", Value::Int64(now)),
                    ]);
                }
            }
        }

        Ok(())
    }

    /// Get items that are frequently co-accessed with the given IDs.
    /// Returns (neighbor_id, co_access_count) tuples.
    pub fn get_co_accessed(&self, ids: &[&str], min_count: i64) -> Result<Vec<(String, i64)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn()?;
        let mut results = Vec::new();

        for id in ids {
            let mut stmt = conn.prepare(
                "MATCH (a:Memory {id: $id})-[r:CO_ACCESSED]-(b:Memory)
                 WHERE r.count >= $min_count
                 RETURN b.id, r.count
                 ORDER BY r.count DESC"
            ).map_err(|e| SedimentError::Database(format!("Failed to prepare co-access query: {}", e)))?;

            let result = conn.execute(&mut stmt, vec![
                ("id", Value::String(id.to_string())),
                ("min_count", Value::Int64(min_count)),
            ]).map_err(|e| SedimentError::Database(format!("Failed to query co-access: {}", e)))?;

            for row in result {
                if let (Some(bid), Some(count)) = (extract_string(&row, 0), extract_i64(&row, 1)) {
                    results.push((bid, count));
                }
            }
        }

        // Deduplicate by target_id, keeping highest count
        results.sort_by(|a, b| b.1.cmp(&a.1));
        let mut seen = std::collections::HashSet::new();
        results.retain(|(id, _)| seen.insert(id.clone()));

        Ok(results)
    }

    /// Transfer all edges from one node to another (used during consolidation merge).
    pub fn transfer_edges(&self, from_id: &str, to_id: &str) -> Result<()> {
        let conn = self.conn()?;

        // Get all RELATED edges from the old node
        let mut stmt = conn.prepare(
            "MATCH (a:Memory {id: $from_id})-[r:RELATED]-(b:Memory)
             WHERE b.id <> $to_id
             RETURN b.id, r.strength, r.rel_type, r.created_at"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare transfer query: {}", e)))?;

        let result = conn.execute(&mut stmt, vec![
            ("from_id", Value::String(from_id.to_string())),
            ("to_id", Value::String(to_id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to query edges for transfer: {}", e)))?;

        let mut edges_to_create: Vec<(String, f64, String, i64)> = Vec::new();
        for row in result {
            if let (Some(bid), Some(str_val), Some(rt), Some(ts)) = (
                extract_string(&row, 0),
                extract_double(&row, 1),
                extract_string(&row, 2),
                extract_i64(&row, 3),
            ) {
                edges_to_create.push((bid, str_val, rt, ts));
            }
        }

        // Create edges on the new node
        for (bid, strength, rel_type, _) in &edges_to_create {
            let _ = self.add_related_edge(to_id, bid, *strength, rel_type);
        }

        Ok(())
    }

    /// Detect triangles of RELATED items (for clustering).
    /// Returns sets of 3 item IDs that form triangles.
    pub fn detect_clusters(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn()?;

        let result = conn.query(
            "MATCH (a:Memory)-[:RELATED]-(b:Memory)-[:RELATED]-(c:Memory)-[:RELATED]-(a)
             WHERE a.id < b.id AND b.id < c.id
             RETURN a.id, b.id, c.id
             LIMIT 50"
        ).map_err(|e| SedimentError::Database(format!("Failed to detect clusters: {}", e)))?;

        let mut clusters = Vec::new();
        for row in result {
            if let (Some(a), Some(b), Some(c)) = (
                extract_string(&row, 0),
                extract_string(&row, 1),
                extract_string(&row, 2),
            ) {
                clusters.push((a, b, c));
            }
        }

        Ok(clusters)
    }

    /// Get full connection info for an item (all edge types).
    pub fn get_full_connections(&self, item_id: &str) -> Result<Vec<ConnectionInfo>> {
        let conn = self.conn()?;
        let mut connections = Vec::new();

        // RELATED edges
        let mut stmt = conn.prepare(
            "MATCH (a:Memory {id: $id})-[r:RELATED]-(b:Memory)
             RETURN b.id, r.rel_type, r.strength"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare connections query: {}", e)))?;

        let result = conn.execute(&mut stmt, vec![
            ("id", Value::String(item_id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to query connections: {}", e)))?;

        for row in result {
            if let (Some(bid), Some(rt), Some(str_val)) = (
                extract_string(&row, 0),
                extract_string(&row, 1),
                extract_double(&row, 2),
            ) {
                connections.push(ConnectionInfo {
                    target_id: bid,
                    rel_type: rt,
                    strength: str_val,
                    count: None,
                });
            }
        }

        // SUPERSEDES edges
        let mut stmt2 = conn.prepare(
            "MATCH (a:Memory {id: $id})-[r:SUPERSEDES]-(b:Memory)
             RETURN b.id"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare supersedes query: {}", e)))?;

        let result2 = conn.execute(&mut stmt2, vec![
            ("id", Value::String(item_id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to query supersedes: {}", e)))?;

        for row in result2 {
            if let Some(bid) = extract_string(&row, 0) {
                connections.push(ConnectionInfo {
                    target_id: bid,
                    rel_type: "supersedes".to_string(),
                    strength: 1.0,
                    count: None,
                });
            }
        }

        // CO_ACCESSED edges
        let mut stmt3 = conn.prepare(
            "MATCH (a:Memory {id: $id})-[r:CO_ACCESSED]-(b:Memory)
             RETURN b.id, r.count"
        ).map_err(|e| SedimentError::Database(format!("Failed to prepare co-access query: {}", e)))?;

        let result3 = conn.execute(&mut stmt3, vec![
            ("id", Value::String(item_id.to_string())),
        ]).map_err(|e| SedimentError::Database(format!("Failed to query co-access: {}", e)))?;

        for row in result3 {
            if let (Some(bid), Some(count)) = (extract_string(&row, 0), extract_i64(&row, 1)) {
                connections.push(ConnectionInfo {
                    target_id: bid,
                    rel_type: "co_accessed".to_string(),
                    strength: 0.0,
                    count: Some(count),
                });
            }
        }

        Ok(connections)
    }

    /// Get the edge count for an item (total number of edges of all types).
    pub fn get_edge_count(&self, item_id: &str) -> Result<u32> {
        let conns = self.get_full_connections(item_id)?;
        Ok(conns.len() as u32)
    }
}

// ==================== Value extraction helpers ====================

fn extract_string(row: &[Value], idx: usize) -> Option<String> {
    row.get(idx).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn extract_double(row: &[Value], idx: usize) -> Option<f64> {
    row.get(idx).and_then(|v| match v {
        Value::Double(d) => Some(*d),
        Value::Float(f) => Some(*f as f64),
        _ => None,
    })
}

fn extract_i64(row: &[Value], idx: usize) -> Option<i64> {
    row.get(idx).and_then(|v| match v {
        Value::Int64(i) => Some(*i),
        Value::Int32(i) => Some(*i as i64),
        _ => None,
    })
}
