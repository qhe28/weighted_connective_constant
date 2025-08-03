# lattices/base.py
import sympy
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Lattice:
    """A container for lattice-specific properties."""
    name: str
    dim: int
    neighbors: List[Tuple[int, ...]]
    symbols: List[sympy.Symbol]
    _memo_canonical: dict = field(default_factory=dict, repr=False)

    def get_canonical_form(self, walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError("This should be implemented by a specific lattice.")

    def get_step_weight(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        raise NotImplementedError("This should be implemented by a specific lattice.")