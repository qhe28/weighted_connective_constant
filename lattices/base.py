import sympy
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Lattice:
    """A container for lattice-specific properties."""
    name: str
    dim: int
    symbols: List[sympy.Symbol]
    vertex_reps: List[Tuple[int, ...]] = field(default_factory=lambda: [(0,)*2])
    _memo_canonical: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.vertex_reps == [(0,)*2] and self.dim > 0:
            self.vertex_reps = [tuple([0] * self.dim)]

    def get_canonical_form(self, walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError("This should be implemented by a specific lattice.")

    def get_step_weight(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        raise NotImplementedError("This should be implemented by a specific lattice.")

    def get_neighbor_vectors(self, node: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Returns the list of valid neighbor vectors from a given node."""
        raise NotImplementedError("This should be implemented by a specific lattice.")
