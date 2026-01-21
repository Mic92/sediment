# Alecto

Semantic memory for AI agents. Local-first, MCP-native.

## Install

```bash
# Via Homebrew
brew tap rendro/tap
brew install alecto

# Or from source
cargo install --path .
```

## Setup

Add Alecto to your MCP configuration:

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alecto": {
      "command": "alecto"
    }
  }
}
```

### Claude Code

Run `alecto init` in your project, or add to `~/.claude/CLAUDE.md`:

```markdown
## Alecto Memory System

Use the Alecto MCP tools for persistent memory storage.

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
alecto           # Start MCP server
alecto init      # Set up Claude Code integration
alecto stats     # Show database statistics
alecto list      # List stored items
```

## How It Works

- **Storage**: LanceDB (embedded, no server)
- **Embeddings**: all-MiniLM-L6-v2 (local, no API keys)
- **Data**: `~/.alecto/data/`

Everything runs locally. Your data never leaves your machine.
