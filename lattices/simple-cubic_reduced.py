import sympy
from typing import Tuple
from .base import Lattice

LATTICE_NAME = 'simple-cubic_reduced'

def get_lattice() -> Lattice:
    """Returns a configured simple-cubic Lattice object with x=y."""
    lattice = Lattice(
        name=LATTICE_NAME,
        dim=3,
        neighbors=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],
        symbols=[sympy.Symbol('x'), sympy.Symbol('z')]
    )

    def get_canonical_form(walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        if walk in lattice._memo_canonical: return lattice._memo_canonical[walk]
        start_node = walk[0]
        translated = tuple((p[0]-start_node[0], p[1]-start_node[1], p[2]-start_node[2]) for p in walk)
        symmetries = [
            translated,
            tuple((-px, py, pz) for px, py, pz in translated),
            tuple((px, -py, pz) for px, py, pz in translated),
            tuple((-px, -py, pz) for px, py, pz in translated),
            tuple((px, py, -pz) for px, py, pz in translated),
            tuple((-px, py, -pz) for px, py, pz in translated),
            tuple((px, -py, -pz) for px, py, pz in translated),
            tuple((-px, -py, -pz) for px, py, pz in translated),
            
            tuple((py, px, pz) for px, py, pz in translated),
            tuple((-py, px, pz) for px, py, pz in translated),
            tuple((py, -px, pz) for px, py, pz in translated),
            tuple((-py, -px, pz) for px, py, pz in translated),
            tuple((py, px, -pz) for px, py, pz in translated),
            tuple((-py, px, -pz) for px, py, pz in translated),
            tuple((py, -px, -pz) for px, py, pz in translated),
            tuple((-py, -px, -pz) for px, py, pz in translated)
        ]
        canonical = min(symmetries)
        lattice._memo_canonical[walk] = canonical
        return canonical

    def get_step_weight(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        if p1[2] != p2[2]: 
            return lattice.symbols[1]
        return lattice.symbols[0]

    lattice.get_canonical_form = get_canonical_form
    lattice.get_step_weight = get_step_weight
    return lattice