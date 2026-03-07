from typing import Any


def rgetattr(obj: Any, attr_path: str) -> Any:
    """Get a nested attribute using dot-separated path (e.g. ``rgetattr(obj, "a.b.c")``)."""
    attrs = attr_path.split(".")
    current = obj

    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)

    return current


def rsetattr(obj: Any, attr_path: str, value: Any) -> None:
    """Set a nested attribute using dot-separated path (e.g. ``rsetattr(obj, "a.b.c", val)``)."""
    if "." not in attr_path:
        return setattr(obj, attr_path, value)
    attr_path, attr = attr_path.rsplit(".", 1)
    obj = rgetattr(obj, attr_path)
    setattr(obj, attr, value)
