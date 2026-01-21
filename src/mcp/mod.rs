//! MCP (Model Context Protocol) server implementation
//!
//! Provides stdio-based JSON-RPC server for LLM integration.

mod protocol;
mod server;
mod tools;

pub use server::run;
