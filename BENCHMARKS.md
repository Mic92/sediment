# Benchmarks

Recall latency benchmarks at various database sizes, with and without graph features (backfill, expansion, co-access, decay scoring).

Run benchmarks yourself:

```bash
cargo bench
```

## Results

Measured on Apple M1 Pro. Times are per-recall (single query, top-5 results).

| DB Size | Graph Off | Graph On | Overhead |
|---------|-----------|----------|----------|
| 100     | ~8ms      | ~10ms    | ~25%     |
| 1,000   | ~12ms     | ~15ms    | ~25%     |
| 10,000  | ~18ms     | ~23ms    | ~28%     |

### What's measured

- **Graph Off**: Pure vector search + result formatting
- **Graph On**: Vector search + decay scoring + graph backfill + 1-hop expansion + co-access suggestions

### Key takeaways

- Sub-25ms recall at 10K items with full graph features
- Graph overhead is ~25-28% regardless of database size
- Embedding computation dominates latency (model inference is ~5ms per query)
- All operations are local — no network round-trips

## Methodology

- Criterion.rs with 20 samples, 10s measurement time
- Databases seeded with mixed content (short text, markdown, code)
- ~30% of items have graph edges
- Queries rotate through 5 representative search terms
- Fresh DB connection per iteration (matches real-world MCP usage)
