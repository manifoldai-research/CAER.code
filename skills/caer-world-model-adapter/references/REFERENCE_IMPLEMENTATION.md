# Framework-neutral reference

The pseudocode is independent of PyTorch, JAX, and any particular diffusion
library. `valid` has shape `[B, *token_axes]`; prediction channels are the
leading non-batch axis.

```python
def caer_loss(model, noisy, target, timestep, action, null_action,
              valid, action_active, eps=1e-6):
    with inference_mode():
        p_on = model(noisy, timestep=timestep, action=action)
        p_null = model(noisy, timestep=timestep, action=null_action)
        effect = l2_norm(stop_gradient(p_on - p_null), channel_axis=1)

    valid_f = valid.astype(effect.dtype)
    effect_sum = (effect * valid_f).sum(token_axes, keepdims=True)
    token_count = valid_f.sum(token_axes, keepdims=True)
    mean_effect = effect_sum / maximum(token_count, 1)
    rho = effect / maximum(mean_effect, eps)
    rho = where(mean_effect > eps, rho, ones_like(rho))
    rho = where(action_active[..., None], rho, ones_like(rho))
    rho = stop_gradient(rho)

    prediction = model(noisy, timestep=timestep, action=action)
    sq_error = (prediction.float() - target.float()) ** 2
    rho_channels = broadcast_to_prediction_channels(rho, sq_error)
    valid_channels = broadcast_to_prediction_channels(valid_f, sq_error)
    numerator = (sq_error * rho_channels * valid_channels).sum(all_nonbatch_axes)
    denominator = (rho_channels * valid_channels).sum(all_nonbatch_axes)
    return (numerator / maximum(denominator, eps)).mean()
```

Adapt the broadcast expression to the repository's tensor layout; do not copy
it literally. This repository uses `[B,C,T,H,W]` and excludes the first frame
from future-token reductions where required. The graph boundary is essential:
`p_on`, `p_null`, `effect`, and `rho` must not contribute gradients, while the
main `prediction` must retain its graph.
