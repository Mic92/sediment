# Sediment

Semantic memory for AI agents. Local-first, MCP-native.

Combines vector search, a relationship graph, and access tracking into a unified memory intelligence layer — all running locally as a single binary.

## Install

```bash
# Via Homebrew
brew tap rendro/tap
brew install sediment

# Or from source
cargo install --path .
```

## Setup

Add Sediment to your MCP configuration:

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### Claude Code

Run `sediment init` in your project, or add to `~/.claude/CLAUDE.md`:

```markdown
## Sediment Memory System

Use the Sediment MCP tools for persistent memory storage.

- `store` - Store content for later retrieval
- `recall` - Search by semantic similarity
- `list` - List stored items
- `forget` - Delete an item by ID
- `connections` - Show relationship graph for an item
```

## Tools

| Tool | Description |
|------|-------------|
| `store` | Save content with optional title, tags, metadata, expiration, scope, replace, and related item links |
| `recall` | Search memories by semantic similarity with decay scoring, trust weighting, graph expansion, and co-access suggestions |
| `list` | List stored items by scope (project/global/all) with tag filtering |
| `forget` | Delete an item by ID (removes from vector store and graph) |
| `connections` | Show relationship graph for an item (related, supersedes, co-accessed edges) |

## CLI

```bash
sediment           # Start MCP server
sediment init      # Set up Claude Code integration
sediment stats     # Show database statistics
sediment list      # List stored items
```

## How It Works

### Three-Database Hybrid

All local, embedded, zero config:

- **LanceDB** — Vector embeddings and semantic similarity search
- **SQLite** (graph) — Relationship tracking: RELATED, SUPERSEDES, CO_ACCESSED, CLUSTER_SIBLING edges
- **SQLite** (access) — Mutable counters: access tracking, decay scoring, consolidation queue

### Key Features

- **Memory decay**: Results re-ranked by freshness (30-day half-life) and access frequency. Old memories rank lower but are never auto-deleted.
- **Trust-weighted scoring**: Validated and well-connected memories score higher.
- **Project scoping**: Automatic context isolation between projects. Same-project items get a similarity boost.
- **Relationship graph**: Items linked via RELATED, SUPERSEDES, and CO_ACCESSED edges. Recall expands results with 1-hop graph neighbors and co-access suggestions.
- **Background consolidation**: Near-duplicates (≥0.95 similarity) auto-merged; similar items (0.85–0.95) linked.
- **Auto-tagging**: Items without tags inherit tags from similar existing items.
- **Type-aware chunking**: Intelligent splitting for markdown, code, JSON, YAML, and plain text.
- **Conflict detection**: Items with ≥0.85 similarity flagged on store.
- **Cross-project recall**: Results from other projects flagged with provenance metadata.
- **Local embeddings**: all-MiniLM-L6-v2 via Candle (384-dim vectors, no API keys).

### Data Location

- Vector store: `~/.sediment/data/`
- Graph + access tracking: `~/.sediment/access.db`

Everything runs locally. Your data never leaves your machine.
