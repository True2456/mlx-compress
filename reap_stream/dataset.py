from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_prompt_texts(dataset_file: str | Path, limit: Optional[int] = None) -> list[str]:
    """Load calib prompts from JSONL with `text` or `messages` fields."""
    path = Path(dataset_file)
    out: list[str] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if not text and rec.get("messages"):
                parts = []
                for m in rec["messages"]:
                    role = str(m.get("role", "?")).upper()
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = "\n".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    parts.append(f"{role}:\n{content}")
                text = "\n\n".join(parts)
            if text and str(text).strip():
                out.append(str(text).strip())
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"No usable prompts in {path}")
    return out
