import sympy
from typing import Tuple, List
from .base import Lattice

LATTICE_NAME = 'hexagonal'

def get_lattice() -> Lattice:
    """Returns a correctly configured hexagonal Lattice object."""
    
    lattice = Lattice(
        name=LATTICE_NAME,
        dim=2,
        symbols=[sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')],
        vertex_reps=[(0, 0), (1, 0)] # K=2
    )

    # Class 0 (like (0,0)): (px+py)%3 == 0
    _NEIGHBORS_CLASS_0 = [(-1, -1), (1, 0), (0, 1)]
    # Class 1 (like (1,0)): (px+py)%3 == 1
    _NEIGHBORS_CLASS_1 = [(-1, 0), (1, 1), (0, -1)]

    def get_neighbor_vectors(node: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Returns the correct list of 3 neighbor vectors based on the node's class.
        """
        class_id = (node[0] + node[1]) % 3
        if class_id == 0:
            return _NEIGHBORS_CLASS_0
        else:
            return _NEIGHBORS_CLASS_1

    def get_canonical_form(walk: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        """
        Calculates the canonical form of a walk by applying all weight-preserving
        symmetries and selecting the lexicographically smallest result.
        
        Symmetries include:
        1. Translation (handled by normalizing to the class origin).
        2. Point reflection (x, y) -> (1-x, -y), which swaps vertex classes.
        """
        if walk in lattice._memo_canonical:
            return lattice._memo_canonical[walk]
        
        # --- Candidate 1: the original walk, translated to its class origin ---
        start_node_orig = walk[0]
        class_id_orig = (start_node_orig[0] + start_node_orig[1]) % 3
        class_origin_orig = lattice.vertex_reps[class_id_orig]
        
        candidate1 = tuple(
            (p[0] - start_node_orig[0] + class_origin_orig[0], p[1] - start_node_orig[1] + class_origin_orig[1])
            for p in walk
        )

        # --- Candidate 2: the reflected walk, translated to its *new* class origin ---
        # Apply the symmetry (px, py) -> (1-px, -py) to each point in the original walk
        symmetric_walk = tuple((1 - p[0], -p[1]) for p in walk)

        start_node_sym = symmetric_walk[0]
        class_id_sym = (start_node_sym[0] + start_node_sym[1]) % 3
        class_origin_sym = lattice.vertex_reps[class_id_sym]

        # Translate this new, symmetric walk to its own class origin
        candidate2 = tuple(
            (p[0] - start_node_sym[0] + class_origin_sym[0], p[1] - start_node_sym[1] + class_origin_sym[1])
            for p in symmetric_walk
        )

        # The canonical form is the lexicographically smallest of all candidates
        canonical = min(candidate1, candidate2)
        
        # Cache the result before returning
        lattice._memo_canonical[walk] = canonical
        return canonical

    def get_step_weight(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> sympy.Symbol:
        step = (p2[0] - p1[0], p2[1] - p1[1])
        if step in [(1, 0), (-1, 0)]:
            return lattice.symbols[0]
        elif step in [(0, 1), (0, -1)]:
            return lattice.symbols[1]
        elif step in [(1, 1), (-1, -1)]:
            return lattice.symbols[2]
        else:
            raise ValueError(f"Invalid step vector {step} on the hexagonal lattice.")

    lattice.get_neighbor_vectors = get_neighbor_vectors
    lattice.get_canonical_form = get_canonical_form
    lattice.get_step_weight = get_step_weight
    return lattice