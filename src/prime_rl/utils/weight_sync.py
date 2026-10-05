WEIGHT_VERSION_HEADER = "X-Prime-Weight-Version"


def generation_weight_error(
    *,
    active_version: str,
    weights_dirty: bool,
    serving_ready: bool,
    require_version: bool,
    minimum_version: str | None,
) -> tuple[int, str] | None:
    if weights_dirty or not serving_ready:
        return 503, "weights are not ready for generation"
    if minimum_version is None:
        if require_version:
            return 400, f"{WEIGHT_VERSION_HEADER} is required"
        return None
    try:
        minimum = int(minimum_version)
    except ValueError:
        return 400, "invalid minimum weight version"
    if minimum < 0:
        return 400, "invalid minimum weight version"
    try:
        active = int(active_version)
    except ValueError:
        return 503, "no synchronized weight version is active"
    if active < minimum:
        return 503, f"active weight version {active_version} is behind requested version {minimum_version}"
    return None
