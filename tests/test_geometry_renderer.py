import numpy as np
from client.renderer.geometry import render_geometry


def test_geometry_output_shape():
    result = render_geometry([], [-5, 5], [-5, 5], 500, 500)
    assert result.shape == (500, 500, 3)
    assert result.dtype == np.uint8


def test_geometry_empty_elements():
    result = render_geometry([], [-5, 5], [-5, 5], 500, 500)
    assert result.shape == (500, 500, 3)


def test_geometry_point():
    elements = [{"type": "point", "pos": [0, 0], "label": "O"}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.sum() > 0


def test_geometry_line():
    elements = [{"type": "line", "from": [-3, 0], "to": [3, 0]}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.sum() > 0


def test_geometry_circle():
    elements = [{"type": "circle", "center": [0, 0], "radius": 3}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.sum() > 0


def test_geometry_circle_dashed():
    elements = [{"type": "circle", "center": [0, 0], "radius": 3, "style": "dashed"}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.shape == (500, 500, 3)


def test_geometry_arc():
    elements = [{"type": "arc", "center": [0, 0], "radius": 2, "start_angle": 0, "end_angle": 90}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.sum() > 0


def test_geometry_with_grid():
    no_grid = render_geometry([], [-5, 5], [-5, 5], 500, 500, show_grid=False)
    with_grid = render_geometry([], [-5, 5], [-5, 5], 500, 500, show_grid=True)
    assert with_grid.sum() > no_grid.sum()


def test_geometry_unknown_element_type_ignored():
    elements = [{"type": "unicorn", "pos": [0, 0]}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.shape == (500, 500, 3)


def test_geometry_custom_colors():
    elements = [{"type": "point", "pos": [0, 0], "color": "#ff0000"}]
    result = render_geometry(elements, [-5, 5], [-5, 5], 500, 500)
    assert result.sum() > 0


def test_geometry_multiple_elements():
    elements = [
        {"type": "point", "pos": [0, 0], "label": "A"},
        {"type": "point", "pos": [3, 4], "label": "B"},
        {"type": "point", "pos": [3, 0], "label": "C"},
        {"type": "line", "from": [0, 0], "to": [3, 4]},
        {"type": "line", "from": [3, 4], "to": [3, 0]},
        {"type": "line", "from": [3, 0], "to": [0, 0]},
    ]
    result = render_geometry(elements, [-2, 6], [-2, 6], 500, 500)
    assert result.sum() > 0
