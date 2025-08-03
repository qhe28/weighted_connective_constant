import os
import importlib
from typing import Dict, Callable
from .base import Lattice

# This dictionary will store the functions that create each lattice.
# e.g., {'square': <function get_lattice from lattices.square>}
_lattice_getters: Dict[str, Callable[[], Lattice]] = {}

def _discover_lattices():
    """Finds all lattice modules in this directory and registers them."""
    if _lattice_getters:  # Discover only once
        return

    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and not filename.startswith(('_', 'base.')):
            module_name = filename[:-3]
            # Import the module dynamically (e.g., 'lattices.square')
            module = importlib.import_module(f'.{module_name}', __name__)
            
            # Register its name and getter function
            lattice_name = getattr(module, 'LATTICE_NAME', None)
            getter_func = getattr(module, 'get_lattice', None)

            if lattice_name and callable(getter_func):
                _lattice_getters[lattice_name] = getter_func

def get_available_lattices() -> Dict[str, Callable[[], Lattice]]:
    """Returns a dictionary of all discovered lattice getters."""
    _discover_lattices()
    return _lattice_getters

def get_lattice(name: str) -> Lattice:
    """Constructs and returns a specific lattice by name."""
    _discover_lattices()
    if name not in _lattice_getters:
        raise ValueError(f"Lattice '{name}' not found. Available: {list(_lattice_getters.keys())}")
    return _lattice_getters[name]()

# Automatically discover lattices when the package is imported.
_discover_lattices()