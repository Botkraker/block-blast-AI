from __future__ import annotations

import pytest

from blockblast.engine.pieces import (
    CATALOGUE_VERSION,
    PIECE_CATALOGUE,
    PLACEMENT_MASKS,
    VALID_PLACEMENTS,
    get_piece,
)


def test_catalogue_v1_has_27_pieces_with_contiguous_ids() -> None:
    assert CATALOGUE_VERSION == 1
    assert len(PIECE_CATALOGUE) == 27
    assert [p.piece_id for p in PIECE_CATALOGUE] == list(range(27))


def test_shapes_are_unique() -> None:
    assert len({p.cells for p in PIECE_CATALOGUE}) == len(PIECE_CATALOGUE)


@pytest.mark.parametrize("piece", PIECE_CATALOGUE, ids=lambda p: p.name)
def test_piece_geometry(piece) -> None:  # type: ignore[no-untyped-def]
    rows = {r for r, _ in piece.cells}
    cols = {c for _, c in piece.cells}
    # bbox is tight and anchored at (0, 0)
    assert min(rows) == 0 and min(cols) == 0
    assert max(rows) + 1 == piece.height and max(cols) + 1 == piece.width
    assert 1 <= piece.size <= 9
    assert piece.height <= 5 and piece.width <= 5
    # 4-connected
    cells = set(piece.cells)
    seen = {piece.cells[0]}
    stack = [piece.cells[0]]
    while stack:
        r, c = stack.pop()
        for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    assert seen == cells


def test_expected_sizes() -> None:
    sizes = {p.name: p.size for p in PIECE_CATALOGUE}
    assert sizes["mono"] == 1 and sizes["I5h"] == 5 and sizes["O2"] == 4 and sizes["O3"] == 9
    assert all(sizes[n] == 4 for n in sizes if n[0] in "LJTSZ")


@pytest.mark.parametrize("piece", PIECE_CATALOGUE, ids=lambda p: p.name)
def test_placement_masks_match_brute_force(piece) -> None:  # type: ignore[no-untyped-def]
    for row in range(8):
        for col in range(8):
            fits = row + piece.height <= 8 and col + piece.width <= 8
            expected = 0
            if fits:
                for dr, dc in piece.cells:
                    expected |= 1 << ((row + dr) * 8 + col + dc)
            assert PLACEMENT_MASKS[piece.piece_id][row * 8 + col] == expected
    valid = VALID_PLACEMENTS[piece.piece_id]
    assert len(valid) == (9 - piece.height) * (9 - piece.width)
    assert all(m for _, m in valid)


def test_get_piece() -> None:
    assert get_piece(0).name == "mono"
    with pytest.raises(IndexError):
        get_piece(27)
    with pytest.raises(IndexError):
        get_piece(-1)
