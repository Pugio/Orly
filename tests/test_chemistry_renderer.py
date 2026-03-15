import numpy as np
from client.renderer.chemistry import render_chemistry, ELEMENT_COLORS


def test_chemistry_output_shape():
    result = render_chemistry([], [], 500, 400)
    assert result.shape == (400, 500, 3)
    assert result.dtype == np.uint8


def test_chemistry_empty_molecule():
    result = render_chemistry([], [], 500, 400)
    # Should be mostly black
    assert result.shape == (400, 500, 3)


def test_chemistry_single_atom():
    atoms = [{"symbol": "O", "pos": [0, 0]}]
    result = render_chemistry(atoms, [], 500, 400)
    assert result.sum() > 0


def test_chemistry_default_element_color():
    # O should be red from ELEMENT_COLORS
    assert "O" in ELEMENT_COLORS
    assert ELEMENT_COLORS["O"] == "#FF0000"


def test_chemistry_custom_element_color():
    atoms = [{"symbol": "O", "pos": [0, 0], "color": "#00ff00"}]
    result = render_chemistry(atoms, [], 500, 400)
    assert result.sum() > 0


def test_chemistry_unknown_element_gets_default():
    atoms = [{"symbol": "Xx", "pos": [0, 0]}]
    result = render_chemistry(atoms, [], 500, 400)
    assert result.shape == (400, 500, 3)


def test_chemistry_single_bond():
    atoms = [{"symbol": "O", "pos": [0, 0]}, {"symbol": "H", "pos": [2, 0]}]
    bonds = [{"from": 0, "to": 1, "order": 1}]
    result = render_chemistry(atoms, bonds, 500, 400)
    assert result.sum() > 0


def test_chemistry_double_bond():
    atoms = [{"symbol": "O", "pos": [0, 0]}, {"symbol": "C", "pos": [2, 0]}]
    bonds = [{"from": 0, "to": 1, "order": 2}]
    result = render_chemistry(atoms, bonds, 500, 400)
    assert result.sum() > 0


def test_chemistry_triple_bond():
    atoms = [{"symbol": "N", "pos": [0, 0]}, {"symbol": "N", "pos": [2, 0]}]
    bonds = [{"from": 0, "to": 1, "order": 3}]
    result = render_chemistry(atoms, bonds, 500, 400)
    assert result.sum() > 0


def test_chemistry_water_molecule():
    atoms = [
        {"symbol": "O", "pos": [0, 0], "color": "#ff0000"},
        {"symbol": "H", "pos": [-1, -0.5]},
        {"symbol": "H", "pos": [1, -0.5]},
    ]
    bonds = [
        {"from": 0, "to": 1, "order": 1},
        {"from": 0, "to": 2, "order": 1},
    ]
    result = render_chemistry(atoms, bonds, 500, 400)
    assert result.sum() > 0


def test_chemistry_with_title():
    atoms = [{"symbol": "O", "pos": [0, 0]}]
    no_title = render_chemistry(atoms, [], 500, 400, title="")
    with_title = render_chemistry(atoms, [], 500, 400, title="Water (H₂O)")
    # Title adds text pixels
    assert with_title.sum() >= no_title.sum()


def test_chemistry_invalid_bond_index_ignored():
    atoms = [{"symbol": "O", "pos": [0, 0]}, {"symbol": "H", "pos": [1, 0]}]
    bonds = [{"from": 0, "to": 5, "order": 1}]  # index 5 doesn't exist
    result = render_chemistry(atoms, bonds, 500, 400)
    assert result.shape == (400, 500, 3)
