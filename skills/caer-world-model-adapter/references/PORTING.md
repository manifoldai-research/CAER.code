# Porting CAER to a new world model

This guide is for an agent modifying an unfamiliar training repository. It
assumes an action-conditioned predictor `f_theta(x_t, c, a)` and a native null
condition `a_null`.

## 1. Freeze the original contract

Record the current noisy input, target, timestep/sigma representation,
scheduler weighting, action dropout, valid-token/padding/frame masks, auxiliary
losses, gradient scaling, and optimizer step. CAER must wrap the existing
prediction error; do not change the target or scheduler while porting.

## 2. Define the causal pair

```text
P_on   = f_theta(noisy_input, timestep, all_conditions, action)
P_null = f_theta(noisy_input, timestep, all_conditions, null_action)
S      = norm(P_on - P_null, axis=prediction_channels)
```

The diagnostic calls share the same noisy input, timestep/sigma, sequence
length, text/reference conditions, parameters, and random decisions. Change
only the action condition. Replay or isolate RNG if the model has stochastic
layers. Diagnostics run in inference/no-grad mode and `S` is detached.

## 3. Align the map with the loss

Reduce `S` to the same token lattice as the squared prediction error. A video
implementation may use `[B, 1, T, H, W]`; an image/sequence model may use
`[B, 1, ...token_axes]`. Broadcast only across prediction channels.

Let `V` be the boolean valid-token mask after all frame/padding rules. Compute
per-sample means only over `V` (and usually future tokens if the first frame is
not predicted), using a positive epsilon. A zero-effect or empty-valid set
falls back to `rho=1`.

The canonical CAER weight is a detached normalized effect map:

```text
S_hat[b,u] = S[b,u] / mean_valid(S[b])
rho[b,u]   = S_hat[b,u].detach()
```

Some validated experiments clamp the normalized effect from below at one before
the final loss. Preserve that policy explicitly when reproducing an experiment.

## 4. Preserve the reduction

For squared error `e[b,k,u]` and channel-broadcast `rho[b,u]`, use:

```text
num[b] = sum_{k,u} V[b,u] * rho[b,u] * e[b,k,u]
den[b] = sum_{k,u} V[b,u] * rho[b,u]
loss   = mean_b(num[b] / max(den[b], eps))
```

Normalize per sample, never across the batch. Keep native timestep weighting
either inside `e` or as a documented multiplier. If action conditioning is
    dropped for a sample, set its entire `rho` map to one. Expose `MSE` and
`CAER` paths without changing checkpoint, data, or optimizer
contracts.

## 5. Test before scaling up

Test shape/broadcast errors, zero and constant effect maps, detached diagnostics
and weights, masked-token exclusion from numerator and denominator, inactive
sample fallback, per-sample normalization, CAER-off equivalence to the original
loss, and one tiny real-model forward/backward plus optimizer step. Only then
wire the adapter into multi-GPU or long-running launch scripts.
