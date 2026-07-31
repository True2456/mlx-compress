"""Strip the tokenizer_class/fix_mistral_regex declarations that make
transformers silently discard Step-3.7's real pretokenizer.

ROOT CAUSE (2026-07-26, supersedes the "no bug, intentional" conclusion in
docs/TOKENIZER-INVESTIGATION.md -- that conclusion is WRONG, see the doc's
correction). Proven three ways:

  1. StepFun's genuinely-shipped tokenizer.json (HF cache, untouched),
     loaded directly via `tokenizers.Tokenizer.from_file()` (no transformers
     involved at all): round-trips PERFECTLY on English, Chinese, and
     digit-heavy strings. '18452' -> 2 tokens ['184','52']. Not broken.
  2. The SAME file, loaded via `AutoTokenizer.from_pretrained()` (what
     mlx_lm.tokenizer_utils.load() -- i.e. LM Studio -- actually calls):
     BROKEN. 'hello world' -> 'helloworld', Chinese -> ''.
  3. The SAME file, wrapped in a bare `PreTrainedTokenizerFast` (bypassing
     class-name dispatch): correct again.

The only variable between (2) and (3) is which Python class transformers
instantiates. tokenizer_config.json declares `"tokenizer_class":
"LlamaTokenizerFast"`. AutoTokenizer sees that KNOWN architecture name and
applies LLAMA's own hardcoded SentencePiece/Metaspace conversion recipe,
discarding tokenizer.json's actual custom 4-stage pretokenizer Sequence
entirely and replacing it with `Metaspace(replacement="▁", split=False)` --
which silently eats spaces on raw (non-SentencePiece-prefixed) input and
returns nothing for Chinese. This has nothing to do with `fix_mistral_regex`
(that flag's own detection is ALSO a false positive here -- Step-3.7's
config.json has `model_type: "step3p7"`, not a Mistral family type, and only
matched because `transformers_version` is absent from config.json -- but
fixing that flag alone does not fix the corruption; the tokenizer_class
misdispatch is the actual cause, verified independently of it).

Fix: delete `tokenizer_class` (and `fix_mistral_regex`, which is irrelevant
once the real cause is gone) from tokenizer_config.json. With no declared
class name, AutoTokenizer falls back to the generic `TokenizersBackend`
class, which trusts tokenizer.json's own pretokenizer as authoritative.

Verified end-to-end, live generation, LM Studio endpoint, temp 0, all
sampler fields pinned: the canonical adversarial-context repro
('kill process ID 18452' in a MAC-address-dense context) that failed 0/12
all investigation -- 5/5 after this fix. A second canonical case (echoing
'2456, 1337, 495' in the same context) -- 5/5 after this fix, also
previously unresolved. No model/weight change; this is a two-key deletion
in one JSON file.

Usage:
    .venv/bin/python scripts/fix_tokenizer_class.py <checkpoint_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fix_tokenizer_class(out_dir: Path) -> None:
    """Strip the offending keys and verify via a real AutoTokenizer round-trip
    through the exact call mlx_lm/LM Studio uses. Raises if verification fails
    -- never leave a build silently wrong."""
    cfg_path = Path(out_dir) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    removed = {k: cfg.pop(k, None) for k in ("tokenizer_class", "fix_mistral_regex")}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[fix_tokenizer_class] removed {removed} from {cfg_path}")

    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out_dir), trust_remote_code=True)
    tests = ["hello world", "你好世界", "中文 with english", "18452",
             "kill -9 18452", "2456, 1337, 495", "float32", "bfloat16"]
    bad = []
    for s in tests:
        ids = tok.encode(s, add_special_tokens=False)
        rt = tok.decode(ids)
        if rt != s:
            bad.append((s, rt))
    if bad:
        raise RuntimeError(
            f"tokenizer still broken after fix, resolved class={type(tok).__name__}: {bad}")
    print(f"[fix_tokenizer_class] verified OK, resolved class={type(tok).__name__} "
          f"({len(tests)}/{len(tests)} round-trips clean)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: fix_tokenizer_class.py <checkpoint_dir>")
        raise SystemExit(2)
    fix_tokenizer_class(Path(sys.argv[1]))
