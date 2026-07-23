# Local patches on Cerebras REAP

Applied for MiniMax-M2 / M2.7 layer-wise REAP:

1. `src/reap/model_util.py` — `MiniMaxM2ForCausalLM` MODEL_ATTRS
2. `src/reap/observer.py` — `MiniMaxM2MoEObserverHookConfig`, sigmoid + bias routing
3. `src/reap/pruning_metrics.py` — `router_score_fn` (`softmax` | `sigmoid`)
4. `src/reap/layerwise_observer.py` — MiniMax top-k via sigmoid + `e_score_correction_bias`
5. `src/reap/prune.py` — preserve `MiniMaxM2Experts`, slice MoE-level bias

Re-apply after `git pull` in this vendor tree.
