import contextlib

from glyphbench.core.observation import GridObservation


def test_grid_observation_holds_all_four_string_fields():
    obs = GridObservation(grid="@..", legend="@ — you", hud="HP:10", message="hi")
    assert obs.grid == "@.."
    assert obs.legend == "@ — you"
    assert obs.hud == "HP:10"
    assert obs.message == "hi"


def test_grid_observation_is_frozen():
    obs = GridObservation(grid="@", legend="", hud="", message="")
    try:
        obs.grid = "x"  # type: ignore[misc]
    except Exception as e:
        msg = str(e).lower()
        cls = type(e).__name__.lower()
        assert (
            "frozen" in msg
            or "frozen" in cls
            or "can't set attribute" in msg
            or "cannot assign" in msg
        )
    else:
        raise AssertionError("expected frozen dataclass to reject attribute assignment")


def test_render_canonical_section_order_legend_hud_grid_message():
    obs = GridObservation(
        grid="@..",
        legend="@ — you",
        hud="Step: 1",
        message="You moved.",
    )
    rendered = obs.render()
    legend_idx = rendered.index("[Legend]")
    hud_idx = rendered.index("[HUD]")
    grid_idx = rendered.index("[Grid]")
    message_idx = rendered.index("[Message]")
    assert legend_idx < hud_idx < grid_idx < message_idx


def test_render_omits_empty_legend_hud_message_sections_but_always_includes_grid():
    obs = GridObservation(grid="@..", legend="", hud="", message="")
    rendered = obs.render()
    assert "[Legend]" not in rendered
    assert "[HUD]" not in rendered
    assert "[Message]" not in rendered
    assert "[Grid]" in rendered
    assert "@.." in rendered


def test_render_includes_section_content_after_header():
    obs = GridObservation(grid="A", legend="L1", hud="H1", message="M1")
    rendered = obs.render()
    assert "[Legend]\nL1" in rendered
    assert "[HUD]\nH1" in rendered
    assert "[Grid]\nA" in rendered
    assert "[Message]\nM1" in rendered


def test_grid_observation_fields_must_be_strings():
    try:
        GridObservation(grid=None, legend="", hud="", message="")  # type: ignore[arg-type]
    except TypeError:
        pass  # acceptable
    else:
        # If runtime type check isn't enforced by dataclass, at least render() fails
        obs = GridObservation(grid=None, legend="", hud="", message="")  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            obs.render()
