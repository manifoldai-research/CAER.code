import torch


METHOD1_LOSS_VARIANTS = ("MSE", "CAER")


def _future_mean(values, exclude_first_frame):
    future_values = (
        values[:, :, 1:]
        if exclude_first_frame and values.size(2) > 1
        else values
    )
    return future_values.mean(dim=(1, 2, 3, 4), keepdim=True)


def _normalize_token_map(values, eps, exclude_first_frame):
    values_mean = _future_mean(values, exclude_first_frame)
    normalized = values / values_mean.clamp_min(float(eps))
    return torch.where(
        values_mean > float(eps),
        normalized,
        torch.ones_like(normalized),
    )


def method1_focused_flow_loss(
    prediction,
    target,
    effect_map,
    active_mask=None,
    eps=1e-6,
    mse_threshold=0.0,
    exclude_first_frame=True,
    loss_variant="CAER",
):
    if loss_variant not in METHOD1_LOSS_VARIANTS:
        raise ValueError(
            f"loss_variant must be one of {METHOD1_LOSS_VARIANTS}; "
            f"got {loss_variant!r}"
        )
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "prediction and target must share shape [B,C,T,H,W]; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    pred_f = prediction.float()
    target_f = target.float()
    sq_error = (pred_f - target_f).pow(2)
    residual_map = torch.linalg.vector_norm(
        pred_f.detach() - target_f, ord=2, dim=1, keepdim=True
    )

    if loss_variant == "MSE":
        rho = torch.ones_like(residual_map)
    else:
        if effect_map is None:
            raise ValueError(f"effect_map is required for loss_variant={loss_variant!r}")
        effect_f = effect_map.detach().float()
        expected_effect_shape = (prediction.shape[0], 1, *prediction.shape[2:])
        if tuple(effect_f.shape) != expected_effect_shape:
            raise ValueError(
                f"effect_map must have shape {expected_effect_shape}; "
                f"got {tuple(effect_f.shape)}"
            )
        if not bool(torch.isfinite(effect_f).all()):
            raise ValueError("effect_map contains non-finite values")

        rho = _normalize_token_map(
            effect_f, eps, exclude_first_frame
        ).detach()

    if active_mask is not None:
        rho = torch.where(
            active_mask.to(dtype=torch.bool), rho, torch.ones_like(rho)
        )

    if exclude_first_frame and pred_f.size(2) > 1:
        sq_error = sq_error[:, :, 1:]
        rho = rho[:, :, 1:]

    if mse_threshold and mse_threshold > 0:
        residual_abs = (pred_f - target_f).abs()
        if exclude_first_frame and residual_abs.size(2) > 1:
            residual_abs = residual_abs[:, :, 1:]
        sq_error = sq_error * (residual_abs <= mse_threshold).to(sq_error.dtype)

    per_sample_uniform_loss = sq_error.detach().mean(dim=(1, 2, 3, 4))
    uniform_loss = per_sample_uniform_loss.mean()
    weighted_error = sq_error * rho
    per_sample_num = weighted_error.sum(dim=(1, 2, 3, 4))
    per_sample_den = rho.expand_as(sq_error).sum(dim=(1, 2, 3, 4)).clamp_min(float(eps))
    per_sample_loss = per_sample_num / per_sample_den
    return (
        per_sample_loss.mean(),
        rho.detach(),
        uniform_loss.detach(),
        per_sample_loss.detach(),
        per_sample_uniform_loss.detach(),
    )
