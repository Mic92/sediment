# Sediment

Semantic memory for AI agents. Local-first, MCP-native.

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
```

## Tools

| Tool | Description |
|------|-------------|
| `store` | Save content with optional title, tags, and metadata |
| `recall` | Search memories by semantic similarity |
| `list` | List stored items with optional tag filtering |
| `forget` | Delete an item by ID |

## CLI

```bash
sediment           # Start MCP server
sediment init      # Set up Claude Code integration
sediment stats     # Show database statistics
sediment list      # List stored items
```

## How It Works

- **Storage**: LanceDB (embedded, no server)
- **Embeddings**: all-MiniLM-L6-v2 (local, no API keys)
- **Memory decay**: Recent and frequently-accessed memories rank higher in recall. Old memories are never deleted, just ranked lower.
- **Data**: `~/.sediment/data/`

Everything runs locally. Your data never leaves your machine.
