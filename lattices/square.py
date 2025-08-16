import sympy
from typing import Tuple, List
from .base import Lattice

LATTICE_NAME = 'square'

def get_lattice() -> Lattice:
    lattice = Lattice(
        name=LATTICE_NAME,
        dim=2,
        symbols=[sympy.Symbol('x'), sympy.Symbol('y')],
        vertex_reps=[(0, 0)]
    )

    _NEIGHBORS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def get_neighbor_vectors(node: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return _NEIGHBORS

    def get_canonical_form(walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        if walk in lattice._memo_canonical: return lattice._memo_canonical[walk]
        start_node = walk[0]
        translated = tuple((p[0] - start_node[0], p[1] - start_node[1]) for p in walk)
        symmetries = [
            translated,
            tuple((-px, py) for px, py in translated),
            tuple((px, -py) for px, py in translated),
            tuple((-px, -py) for px, py in translated)
        ]
        canonical = min(symmetries)
        lattice._memo_canonical[walk] = canonical
        return canonical
    
    def get_step_weight(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        return lattice.symbols[0] if p1[1] == p2[1] else lattice.symbols[1]

    lattice.get_neighbor_vectors = get_neighbor_vectors
    lattice.get_canonical_form = get_canonical_form
    lattice.get_step_weight = get_step_weight
    return lattice