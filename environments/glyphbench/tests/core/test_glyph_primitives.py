import pytest

from glyphbench.core.glyph_primitives import (
    build_legend,
    draw_box,
    grid_to_string,
    make_empty_grid,
    stamp_sprite,
)


def test_make_empty_grid_default_fill_dot():
    g = make_empty_grid(3, 2)
    assert g == [["\u00b7", "\u00b7", "\u00b7"], ["\u00b7", "\u00b7", "\u00b7"]]


def test_make_empty_grid_custom_fill():
    g = make_empty_grid(2, 2, fill="#")
    assert g == [["#", "#"], ["#", "#"]]


def test_make_empty_grid_rejects_multi_char_fill():
    with pytest.raises(AssertionError):
        make_empty_grid(2, 2, fill="##")


def test_stamp_sprite_in_bounds():
    g = make_empty_grid(3, 3)
    stamp_sprite(g, 1, 1, "@")
    assert g[1][1] == "@"


def test_stamp_sprite_out_of_bounds_raises():
    g = make_empty_grid(3, 3)
    with pytest.raises(IndexError):
        stamp_sprite(g, 5, 5, "@")
    with pytest.raises(IndexError):
        stamp_sprite(g, -1, 0, "@")


def test_grid_to_string_newline_joined():
    g = [["a", "b", "c"], ["d", "e", "f"]]
    assert grid_to_string(g) == "abc\ndef"


def test_draw_box_outline():
    g = make_empty_grid(5, 4)
    draw_box(g, 0, 0, 4, 3)
    s = grid_to_string(g)
    lines = s.split("\n")
    # Top row: + - - - +
    assert lines[0] == "+---+"
    assert lines[-1] == "+---+"
    assert lines[1] == "|\u00b7\u00b7\u00b7|"
    assert lines[2] == "|\u00b7\u00b7\u00b7|"


def test_draw_box_minimal_2x2():
    g = make_empty_grid(2, 2)
    draw_box(g, 0, 0, 1, 1)
    # In a 2x2 box the corners overwrite any edges; all four cells are corners.
    assert g == [["+", "+"], ["+", "+"]]


def test_draw_box_custom_chars():
    g = make_empty_grid(4, 3)
    draw_box(g, 0, 0, 3, 2, horizontal="=", vertical=":", corner="*")
    assert grid_to_string(g) == "*==*\n:\u00b7\u00b7:\n*==*"


def test_draw_box_interior_unchanged():
    g = make_empty_grid(5, 5)
    # Place a sentinel inside the box area
    g[2][2] = "X"
    draw_box(g, 0, 0, 4, 4)
    # Interior sentinel must survive — draw_box only writes the outline.
    assert g[2][2] == "X"


def test_draw_box_rejects_degenerate_coords():
    g = make_empty_grid(4, 4)
    with pytest.raises(ValueError):
        draw_box(g, 2, 2, 2, 2)  # zero area
    with pytest.raises(ValueError):
        draw_box(g, 3, 3, 1, 1)  # inverted


def test_draw_box_rejects_out_of_bounds():
    g = make_empty_grid(4, 4)
    with pytest.raises(IndexError):
        draw_box(g, 0, 0, 4, 3)  # x1=4 is OOB for width 4
    with pytest.raises(IndexError):
        draw_box(g, -1, 0, 3, 3)


def test_build_legend_sorted_deterministic():
    legend = build_legend({"@": "you", "#": "wall", ".": "floor"})
    # Sorted alphabetically by symbol char
    assert legend.index("#") < legend.index(".")
    assert legend.index(".") < legend.index("@")


def test_build_legend_contains_every_entry():
    legend = build_legend({"@": "you", ".": "floor"})
    assert "@" in legend
    assert "you" in legend
    assert "." in legend
    assert "floor" in legend
