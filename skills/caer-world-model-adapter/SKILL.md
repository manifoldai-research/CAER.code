---
name: caer-world-model-adapter
description: Adapt Causal Action Effect Reweighting (CAER) to an action-conditioned diffusion, flow-matching, or sequence world model while preserving the model's original target, scheduler, masking, and optimization semantics.
---

# CAER world-model adapter

Use this skill when a repository trains a world model with an action/control
condition and the user wants action-sensitive, annotation-free loss weighting.
CAER is a training adaptation, not a new model architecture: it measures how
much the prediction changes when the action is removed, detaches that signal,
and uses it to redistribute a fixed per-sample MSE/flow-matching budget.

## Required outcome

Deliver a small, reviewable adapter that:

1. keeps the original prediction target and scheduler/noise parameterization;
2. computes matched on-action and null-action diagnostic predictions;
3. forms a detached causal-effect map `S = ||P_on - P_null||₂` over prediction
   channels (or the equivalent token feature axis);
4. normalizes weights per sample, applies valid-token/conditioning masks, and
   computes the weighted loss without creating gradients through diagnostics;
5. preserves the original loss exactly when CAER is disabled; and
6. includes unit tests plus a real forward/backward and one optimizer-step smoke
   test.

Read [references/PORTING.md](references/PORTING.md) before editing a new
repository. Use [references/REFERENCE_IMPLEMENTATION.md](references/REFERENCE_IMPLEMENTATION.md)
for framework-neutral pseudocode and invariants.

## Operating procedure

- Trace the existing training path first: noisy input construction, target,
  timestep/sigma, condition dropout, masks, auxiliary losses, gradient scaling,
  and optimizer step.
- Define one deterministic null condition using the model's native semantics
  (zero action, dropped action, empty control, or an existing null embedding).
- Run `P_on` and `P_null` with identical noisy input, noise level, shape,
  non-target conditions, parameters, and deterministic/replayed RNG. Change
  only the action condition.
- Run diagnostics under `no_grad`/inference mode and keep them outside the
  gradient-bearing main forward. Detach `S` and every derived weight map.
- Map a requested diagnostic noise level by the model's physical sigma/noise
  coefficient, never by assuming a scheduler-array percentage means the same
  thing across schedulers.
- Normalize over valid future tokens for each sample. If an effect map is zero,
  non-finite, empty after masking, or the condition was dropped, use `rho=1`.
- Apply the same valid-token mask to numerator and denominator. Keep any
  existing timestep weighting and auxiliary objectives unless explicitly asked
  to change them.
- Expose clear `MSE` and `CAER` switches. Do not
  silently change checkpoint, data, or optimizer contracts.

## Review gates

Reject the patch until all of these are demonstrated:

- `P_on` and `P_null` differ only in the action/control condition;
- diagnostic tensors have no gradient path to model parameters;
- `rho` is detached, finite, non-negative, and normalized per sample;
- inactive/dropped samples receive uniform weight;
- masked/padded tokens do not affect either loss normalization term;
- CAER-off produces the pre-adapter loss to numerical tolerance;
- a tiny real batch completes forward, backward, and an optimizer update.

Do not add a second scheduler, architectural wrapper, feature flag system, or
defensive compatibility layer unless the target repository already requires it.
