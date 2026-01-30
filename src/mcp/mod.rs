//! MCP (Model Context Protocol) server implementation
//!
//! Provides stdio-based JSON-RPC server for LLM integration.

mod protocol;
mod server;
pub mod tools;

pub use server::run;
pub use tools::{RecallConfig, RecallResult, recall_pipeline};
