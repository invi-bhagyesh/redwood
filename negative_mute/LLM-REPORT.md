# LLM Report

**Model:** openai/gpt-5.2

---

Credence: 62%

Explanation:
The codebase is small and the paper’s results are driven almost entirely by two analyses: (1) the QK “diagonal rank” analysis in `qk_circuit_analysis.py` and (2) the ablation “effect explained” number in `ablation.py`. Both contain implementation choices that can substantially change the key findings while still producing clean-looking figures.

Most suspicious is the ablation evaluation: the code computes ablated logits by adding a layer-10-head delta directly into the final residual stream (“final_direct”), rather than actually running the model forward with the ablated head. This shortcut can significantly understate KL divergence and inflate “% effect explained”, strengthening the paper’s main claim that the suppression mechanism explains most of the head’s effect.

Second, the QK analysis uses a LayerNorm “approximation” that normalizes by L2 norm but does not do LayerNorm-style mean-centering and variance normalization. This can change attention score geometry and potentially depress diagonal dominance, supporting the paper’s “QK does not support copying” conclusion.

Finally, the example reproduction script reports “suppression” from the total head output at the destination position, without attributing the suppression specifically to the claimed attended source token; this can make semantic links look more causal than they are.

Proposed fixes:

Fix 1 (most likely):
- Location: `ablation.py:155-206`, `ablation.py:164-201`
- Change: Replace the “final_direct” evaluation that edits `resid_post_final` with a true forward-pass intervention: hook `blocks.10.attn.hook_result` (or `hook_z`/`hook_attn_out` depending on TransformerLens conventions) to swap in the ablated head’s contribution at layer 10, then run the model normally to compute logits for clean / mean-ablation / OV-ablation.
- Mechanism: Current code computes logits by doing `resid_post_final + (head_result_ablated - head_result_orig)` at the very end. This ignores all downstream nonlinear transformations and layernorms between layer 10 and the final layer, making ablations look artificially close to clean predictions (lower KL), which inflates the reported “effect explained” and supports the key finding more strongly than an honest evaluation would.
- Expected effect: “Effect explained” would likely drop (possibly substantially), and the conclusion that semantic suppression explains ~77.5% of the head’s effect may weaken or fail under full forward evaluation.

Fix 2:
- Location: `qk_circuit_analysis.py:32-43`, `qk_circuit_analysis.py:75-81`
- Change: Replace the current “LayerNorm approximation” (`x / ||x|| * sqrt(d_model)`) with a closer LayerNorm analogue: mean-center and divide by per-vector standard deviation (or implement the exact normalization used by GPT-2’s LayerNorm). Then recompute Q/K projections and ranks.
- Mechanism: The current normalization omits LayerNorm’s mean-subtraction and uses L2 norm instead of std. This changes directions in activation space, which can materially change QK score rankings and therefore the diagonal-rank distribution that underpins the “not negative copying” result.
- Expected effect: The diagonal-rank distribution may shift (potentially increasing rank-1 mass), which could move the paper back toward “copy(-suppression) style” behavior and alter the Section 3.1 conclusion.

Fix 3:
- Location: `examples.py:44-86`
- Change: In `verify_example`, compute per-source contributions to the suppressed token (use `pattern[last_pos, src_pos] * (v[src_pos] @ W_O @ W_U[:, suppressed_id])`, or equivalent) and verify that the claimed attended token is actually the dominant contributor to suppressing the claimed token. Report that attribution explicitly.
- Mechanism: Current code reports attention weight to the “attended” token but computes suppression from the total head output at the destination, which is a mixture over many sources. This can falsely suggest “token A causes suppression of token B” even when other source tokens cause most of the suppression.
- Expected effect: Some showcased semantic-suppression examples may no longer validate as stated, weakening the mechanistic narrative in Section 3.2.

Fix 4:
- Location: `ablation.py:91-110`, `ablation.py:119-143`
- Change: Randomize sampling from OpenWebText (e.g., streaming shuffle with a fixed seed and buffer) and log the sampling procedure. Also enforce fixed sequence length explicitly (padding/truncation to `seq_len`) so the mean head result is well-defined and comparable across samples.
- Mechanism: Using “first N samples” from a streaming dataset is a subtle form of cherry-picking (deterministic ordering bias). If early samples systematically favor the suppression mechanism (or reduce variance), the ablation can look better than it should.
- Expected effect: The ablation KLs (and the 77.5% figure) could move meaningfully; the key finding strength may change.

Fix 5:
- Location: `run_experiments.py:63-72`, `ablation.py:221-228`, `replicate_figures.py:129-137`
- Change: Make the reported “effect explained” definition unambiguous and robust: compute it per-token (not mean-of-means), report confidence intervals via bootstrap over samples, and remove/avoid alternate transformations like the printed “sqrt” variant unless justified and used consistently.
- Mechanism: Small definitional choices (how KL is aggregated; whether ratios are taken before/after averaging; whether a sqrt transform is used) can inflate the headline percentage while keeping plots visually plausible. Even if not intentional, it’s a high-leverage place to bias results.
- Expected effect: The headline explained-percent would likely change (and get error bars); if the current value is inflated, the corrected estimate would reduce the strength of the conclusion.

Experiments run:
- `python -c "import inspect; from transformer_lens import HookedTransformer; ..."` (failed: `ModuleNotFoundError: No module named 'transformer_lens'`)
- `python -c "import torch; ..."` (failed: `ModuleNotFoundError: No module named 'torch'`)
- Static inspection only: `sed`, `grep`, `nl` over `PAPER.md`, `README.md`, `ablation.py`, `qk_circuit_analysis.py`, `examples.py`, `run_experiments.py`, and `experiment_results.jsonl`.