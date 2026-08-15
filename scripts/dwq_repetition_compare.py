import json, sys, time
import mlx.core as mx
from mlx_vlm import load
from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template
sys.path.insert(0, '.')
from reap_stream.eval_repetition import _distinct_n, _longest_verbatim_repeat

path, label = sys.argv[1], sys.argv[2]
maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 300
model, proc = load(path, lazy=True, strict=False)
lm = model.language_model
tok = getattr(proc, "tokenizer", proc)
_e = getattr(tok, "eos_token_ids", None)
if _e is None: _e = tok.eos_token_id
eos = set(_e) if isinstance(_e, (list, tuple, set)) else {int(_e)}

def gen(ids, n):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    x = mx.array(ids)[None]; out = []
    for _ in range(n):
        o = lm(x, cache=cache)
        logits = (o.logits if hasattr(o, "logits") else o)[:, -1, :]
        nid = int(mx.argmax(logits, axis=-1).item())
        if nid in eos: break
        out.append(nid); x = mx.array([[nid]])
    return out

flagged = 0; n = 0; t0 = time.time()
for line in open('calib/ds4_agentic_repetition_probes.jsonl'):
    r = json.loads(line); m = r['messages']
    pre = m[:-1] if m[-1]['role'] == 'assistant' else m
    rendered = apply_chat_template(pre, add_generation_prompt=True, thinking_mode="chat")
    ids = tok.encode(rendered) if isinstance(rendered, str) else rendered
    g = gen(ids, maxtok)
    d3 = _distinct_n(g, 3); d4 = _distinct_n(g, 4); lr = _longest_verbatim_repeat(g)
    bad = (d4 < 0.5) or (lr >= 20)
    flagged += bad; n += 1
    print(f"  [{'LOOP' if bad else ' ok '}] {r.get('id','?')[:26]:28s} n_gen={len(g):4d} d3={d3:.3f} d4={d4:.3f} rep={lr}", flush=True)
print(f"{label}: {flagged}/{n} loop-suspect   ({time.time()-t0:.0f}s)", flush=True)
