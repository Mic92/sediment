# Changelog

## [0.2.0] - 2025-05-20

### Changed
- Replaced Kuzu graph database with SQLite for relationship tracking
- 5-tool MCP API: `store`, `recall`, `list`, `forget`, `connections`

### Added
- SQLite-based graph store with RELATED, SUPERSEDES, CO_ACCESSED, CLUSTER_SIBLING edges
- Benchmarks for recall latency at various database sizes
- Optimized recall latency: vector index, batch fetch, cached project_id
- Memory decay scoring for recall results
- Background consolidation engine

## [0.1.0] - 2025-05-15

### Added
- Initial release
- Local semantic memory with LanceDB vector storage
- Local embeddings via all-MiniLM-L6-v2 (Candle)
- MCP server with stdio JSON-RPC transport
- Project scoping with automatic isolation
- Type-aware content chunking (markdown, code, JSON, YAML, text)
- CLI: `sediment init`, `sediment stats`, `sediment list`
- Homebrew tap installation
