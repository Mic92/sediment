# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build
cargo build --release

# Run MCP server (default)
cargo run --release

# Run with verbose logging
cargo run --release -- --verbose

# Run CLI commands
cargo run --release -- init          # Initialize project
cargo run --release -- stats         # Show database statistics
cargo run --release -- list          # List stored items

# Run tests (requires model download on first run)
cargo test
cargo test -- --ignored              # Include tests that require model download

# Install locally
cargo install --path .
```

## Architecture

Alecto is a semantic memory system for AI agents, running as an MCP (Model Context Protocol) server.

### Core Components

- **`src/main.rs`** - CLI entry point with subcommands (init, stats, list) and MCP server startup
- **`src/lib.rs`** - Library root exposing public API, project detection, and scope types
- **`src/db.rs`** - LanceDB wrapper handling vector storage, search, and CRUD operations
- **`src/embedder.rs`** - Local embeddings using `all-MiniLM-L6-v2` via Candle (384-dim vectors)
- **`src/chunker.rs`** - Smart content chunking by type (markdown, code, JSON, YAML, text)

### MCP Server (`src/mcp/`)

- **`mod.rs`** - Module exports
- **`server.rs`** - stdio JSON-RPC server implementation with shared embedder context
- **`tools.rs`** - 4 MCP tools: `store`, `recall`, `list`, `forget`
- **`protocol.rs`** - MCP protocol types and JSON-RPC handling

### Data Flow

1. Content → Embedder (generates 384-dim vector) → LanceDB storage
2. Long content (>1000 chars) → Chunker (type-aware splitting) → Individual chunk embeddings
3. Search queries → Embedder → Vector similarity search on both items and chunks → Boosted by project context

### Key Design Decisions

- **Single central database** at `~/.alecto/data/` stores all projects
- **Project scoping** via UUID stored in `.alecto/config` per project
- **Similarity boosting**: Same-project items get 1.15x boost, different projects 0.95x penalty
- **Conflict detection**: Items with >=0.85 similarity flagged on store
- **Fresh DB connection per tool call** with shared embedder for efficiency

## MCP Tools Reference

When working on tools, the 4-tool API is defined in `src/mcp/tools.rs`:

| Tool | Purpose |
|------|---------|
| `store` | Store content with optional title, tags, metadata, expiration, scope |
| `recall` | Semantic search with similarity threshold and tag filtering |
| `list` | List items by scope (project/global/all) with tag filtering |
| `forget` | Delete item by ID |

## Testing Notes

- Embedder tests are marked `#[ignore]` because they require downloading the model (~90MB)
- Use `tempfile` crate for database tests to avoid polluting real data
- LanceDB operations are async; tests use `tokio::runtime::Runtime`
