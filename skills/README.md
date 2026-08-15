# Skills

Agent-ingestible knowledge packaged from this repo's findings.

- `mlx-compress/` — quantization (AWQ/DWQ), bit allocation from importance
  matrices, REAP/REAM, streaming for models larger than unified memory, and the
  evaluation discipline that stops proxies from misleading you.

Install for Claude Code:

```bash
cp -r skills/mlx-compress ~/.claude/skills/
```

Related existing skills, which this one deliberately does not duplicate:
`mlx` (running and fine-tuning), `swift-mlx-lm` (Swift inference),
`mlx-reap-streaming` (REAP saliency collection specifically).
