# Reviewed sparse supervision for RGB-Role-Effect-ICL

This experimental recipe replaces the requirement that **every training pair
provide every auxiliary target**. It does not mark rejected clips as fully
accepted. Missing review records, altered source caches, unreviewed frames,
identity changes, and incompatible demonstration goals still fail validation
or disable the relevant targets. The legacy complete-supervision configuration
and its stricter gate remain available.

## What changed

- `human_valid` / `robot_valid`: reviewed visible regions for mask supervision.
- `robot_effect_valid`: reviewed regions suitable for pooled-feature changes.
- `robot_relation_valid`: reviewed regions suitable for image-plane relative
  centroid changes. A visible fragment need not be a reliable geometry target.
- `cross_effect_valid`: whether human and robot demonstrations express the same
  effect; this also masks the original RGB transfer auxiliary loss.
- The released policy manifest additionally removes seven verified incompatible
  human/robot pairs. Leaving them in behavior cloning while merely disabling
  auxiliary losses would still train the wrong demonstration-to-action mapping.
- `role_future_target_mode=all_reviewed` can use later reviewed targets outside
  the current action chunk, up to the existing 64-frame target horizon. It uses
  the same physical entity at both ends. Future features/masks remain detached
  auxiliary targets and never become policy conditioning inputs.
- Donor transfer still requires explicit single-effect, stable-role semantic
  equivalence. The new package retains 89 eligible recipient pairs; it does not
  invent donors for the remaining pairs.

At deployment, the model uses the human video features and current robot
observation. SAM2, role masks, entity IDs, and robot-future targets are training
supervision, not runtime inputs. The original human Wan context is retained.
ICL language conditioning remains disabled in the existing inference path.

## Annotation release: role-annotations-repaired-20261122

The original 18-part archive and its extracted diagnostic files were verified
against their SHA256 manifests. The archive SHA256 is
`d483cd82a1ab8f0c86d7ee0a6d23fccca2398e3449bc898b69b82674784fa7d1`.

The package contains 600 audit records and 593 recommended training pairs
across the same 12 tasks. All 115 previously rejected pairs received new,
explicit sparse-frame review decisions. Three were inspected over all cached
sampled frames; the other approvals are restricted to the inspected anchors.
One pan sequence was regenerated with SAM2 corrections. The 485 original
accepted reviews are reused with their provenance; this is not a claim that
all 485 clips were independently re-reviewed. Eleven teacher model names were
corrected from the actual SAM2 configuration and weight digest.

These are **SAM2 weak targets reviewed by Codex (AI), not human ground truth**.
Original rejections and evidence remain in the package. Invalid or unseen
regions are not restored and occluded surfaces are not hallucinated.

- `reviewed_manifest.json`: 593 approved policy pairs, portable cache references.
- `icl_manifest.train.json`: the exact matching 593 ICL pairs, without replacement
  demonstrations. All matched baselines must use this selection as well.
- `audit_manifest.all600.json`: historical audit only; training rejects it.
- `labels/`, `reviews/`: masks, identity/scope decisions, source hashes, PNGs.
- `coverage.json`: frame-role coverage and reachable auxiliary target counts.
- `target-motion-audit.json`: change-target diagnostic, not accuracy or success.
- `demos/`: PNG examples and coverage table; no HTML viewer required.
- `reports/`: local preflight, test and short-training/inference evidence.

Coverage remains uneven. Bread-in-pan has object/relationship targets in 16/9
of 50 pairs; cabinet has 7/6. Some accepted bottle/can pairs contain mostly
early static targets. A centroid-shift diagnostic is included, but its absence
does not prove absence of 3-D movement and its presence can reflect occlusion.
This release is a usable sparse-supervision experiment, not complete dense
labels or evidence that the method improves unseen-task success.

## Bind on the training machine

Use the original B300 RGB cache, not a newly extracted or similarly named cache.
The binder verifies exact source bytes and does not rewrite any hash to force a
match. Set these variables for your actual paths:

```bash
export PYTHON_BIN=/path/to/model-env/bin/python
export ROLE_ANNOTATIONS=/path/to/role-annotations-repaired-20261122
export HUMAN_GEN_ROOT=/path/to/HumanGen
export ZERO_WAM_RGB_FEATURE_ROOT=/path/to/rgb_transition_dinov2_base

"$PYTHON_BIN" tools/configure_sparse_role_annotations.py \
  --annotations-dir "$ROLE_ANNOTATIONS" \
  --feature-root "$ZERO_WAM_RGB_FEATURE_ROOT" \
  --human-gen-root "$HUMAN_GEN_ROOT" \
  --config-name robotwin_subset12_role_effect_sparse_b300

export ZERO_WAM_ROLE_MANIFEST="$ROLE_ANNOTATIONS/runtime_manifest.json"
export ZERO_WAM_ICL_MANIFEST="$ROLE_ANNOTATIONS/icl_manifest.train.json"
export MODEL_PATH=/path/to/zero-wam-pretrain
```

Require `pairs=593`, `tasks=12`, and `formal_training_preflight_passed=true`.
Do not run legacy `finalize` on this schema or set a legacy partial flag.
If moving an already bound directory, use a fresh `--runtime-manifest` path;
the binder refuses to silently overwrite a different existing binding.

## Training

Start a fresh experiment from the original Zero-WAM weights. A bounded pipeline
check uses 17 explicit pairs spanning the tasks and the difficult sparse cases:

```bash
export ZERO_WAM_SAVE_ROOT=/path/to/new-sparse-smoke-run
read -r smoke_pairs < "$ROLE_ANNOTATIONS/smoke_pair_keys.txt"
bash script/train_role_effect_sparse.sh --smoke-test \
  --rgb-smoke-sample-ids "$smoke_pairs" --num-steps 20 --save-interval 20 \
  --disable-wandb --load-worker 0 --drop-icl 0
```

For the actual B300 experiment, choose a **different output directory** and omit
all smoke arguments. The default recipe is one GPU, accumulation 8, effective
batch 8, 15,000 updates; it includes the same video/action/MCP objectives:

```bash
export ZERO_WAM_SAVE_ROOT=/path/to/new-sparse-full-run
bash script/train_role_effect_sparse.sh --disable-wandb
```

The uploaded B300 tree has additional full-state resume support. The delivery
package supplies a patch specifically merged against that snapshot so those
changes are retained. On that patched tree, add `--full-state-checkpoint` to
save optimizer/scheduler/RNG state. This option is not available on the plain
upstream-based repair branch. See the package's `B300_HANDOFF.md` before applying
code; do not replace the B300 working tree wholesale or discard local changes.

For four A800 GPUs, set `CONFIG_NAME=robotwin_subset12_role_effect_sparse`.
The launcher then uses four ranks with accumulation 2. Short training verifies
execution and gradients, not convergence, representation quality, or success.

## Experiments needed to support the method

Keep the same 593 pairs, DINO features, reviewed label scopes, future targets,
initial weights, token/parameter budgets and training schedule. Compare the
original RGB-transition model, a resource-matched model with joint rather than
separated representations, and the proposed separated representation. Re-train
ablations; do not rely only on deleting a channel at test time.

In RoboTwin, evaluate the same actual episode seeds and demonstration video
hashes. A seed alone is insufficient if expert feasibility filtering changes
the chosen episodes. Record per-episode outcomes, multiple training seeds, and
paired confidence intervals. Correct-versus-alternative feasible video prompts
in the same initial scene test conditional task following. Zero/shuffled-effect
tokens are useful diagnostics but are distribution shifts, not sole causal
proof. The original human-context path may provide redundant information.

Unseen task, layout and object splits test different generalization claims.
Ordered/repeated-operation claims also need process-level event metrics and
new task variants: a final-state success flag cannot measure every order/count.
Simulator state may be used for evaluation or explicitly declared training
supervision, never silently provided as a policy input. Real-robot experiments
can test domain transfer; they do not remove confounds in these controls.

Related methods already cover important nearby ideas: [ViVLA](https://arxiv.org/abs/2512.07582),
[UniSkill](https://kimhanjung.github.io/UniSkill/),
[XSkill](https://arxiv.org/abs/2307.09955), and
[LTM](https://arxiv.org/abs/2610.21362). Adding DINO, future prediction, memory,
or generic skill composition alone is not a defensible novelty claim.

A possible second contribution is **composition-consistent effect prediction**:
apply the same learned effect transition to the scene produced by the previous
effect, and supervise intermediate as well as final predicted scene features.
For example, placing A on B and then C on A requires the second operation to
refer to A's updated location. This is a proposed experiment, not implemented
by the current two-role delta decoder. A real implementation needs a scene
representation retaining other objects, ordered event segments, and physically
valid composed demonstrations; concatenating unrelated trajectories is invalid.
It should test held-out operation orders and repetition counts with process
metrics. Inference can construct this ordered latent rollout before execution,
without adding reflection or an external replanning loop. The novelty would
need to lie in a specific cross-scene compositional training constraint and its
measured benefit, not the already-established idea of composing skills or
storing transition memory. UniSkill, XSkill and LTM are necessary comparisons.
