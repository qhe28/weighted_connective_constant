import sympy
from typing import Tuple, List
from .base import Lattice

LATTICE_NAME = 'triangular'

def get_lattice() -> Lattice:
    lattice = Lattice(
        name=LATTICE_NAME,
        dim=2,
        symbols=[sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')],
        vertex_reps=[(0, 0)]
    )

    _NEIGHBORS = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]

    def get_neighbor_vectors(node: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return _NEIGHBORS

    def get_canonical_form(walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        if walk in lattice._memo_canonical: return lattice._memo_canonical[walk]
        start_node = walk[0]
        translated = tuple((p[0] - start_node[0], p[1] - start_node[1]) for p in walk)
        symmetries = [
            translated,
            tuple((-px, -py) for px, py in translated)
        ]
        canonical = min(symmetries)
        lattice._memo_canonical[walk] = canonical
        return canonical
    
    def get_step_weight(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        if p1[0] == p2[0]:
            return lattice.symbols[1]
            # return lattice.symbols[0]  # Uncomment this line and comment the above line to weigh x and y steps the same
        elif p1[1] == p2[1]:
            return lattice.symbols[0]
        else:
            return lattice.symbols[2]

    lattice.get_neighbor_vectors = get_neighbor_vectors
    lattice.get_canonical_form = get_canonical_form
    lattice.get_step_weight = get_step_weight
    return lattice