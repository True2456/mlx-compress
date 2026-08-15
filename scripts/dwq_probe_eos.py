import json, sys
import mlx.core as mx
from mlx_vlm import load
from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template
path, label, want = sys.argv[1], sys.argv[2], sys.argv[3]
model, proc = load(path, lazy=True, strict=False)
lm = model.language_model
tok = getattr(proc, "tokenizer", proc)
rec = None
for line in open('calib/ds4_agentic_repetition_probes.jsonl'):
    r = json.loads(line)
    if want in str(r.get('id','')): rec = r; break
msgs = rec['messages']
prefix = msgs[:-1] if msgs[-1].get('role') == 'assistant' else msgs
rendered = apply_chat_template(prefix, add_generation_prompt=True, thinking_mode="chat")
ids = tok.encode(rendered) if isinstance(rendered, str) else rendered
out = lm(mx.array(ids)[None])
logits = (out.logits if hasattr(out,'logits') else out)[0][-1].astype(mx.float32)
probs = mx.softmax(logits, axis=-1)
top = mx.argsort(-probs)[:5].tolist()
print(f"--- {label} / {rec.get('id')} ---")
for t in top:
    print(f"   id={t:<6} p={float(probs[t].item()):.4f} EOS={t==1} {tok.decode([t])!r}")
print(f"   P(EOS)={float(probs[1].item()):.4f}")
