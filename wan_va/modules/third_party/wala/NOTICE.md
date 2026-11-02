# WALA source attribution

`latent_action_model.py` is an unmodified copy of
`wala/model/modules/latent_action_model/latent_action_model.py` from:

- Repository: https://github.com/liujiahao2077/WALA
- Commit: `4a944f07afe09e24743f20d3d9ab809db672dc63`
- Copyright: Copyright (c) 2026 WALA authors.
- License: MIT; the original license is included in `LICENSE`.

The surrounding `wan_va.modules.rgb_transition` implementation is a new
integration. It always constructs this tokenizer with `use_depth=False`; no
depth model, depth data, or released WALA weights are required. WALA's original
RGB objective is retained: mean absolute feature-delta error plus cosine loss.
The upstream argument named `smooth_l1_weight` weights an L1 loss in this
revision; the integration calls this quantity `l1_weight` to avoid ambiguity.

RGB feature normalization, clip order embeddings, human-to-robot transfer
supervision, and gated fusion are integration choices, not claims about the
original WALA method or checkpoint compatibility.
