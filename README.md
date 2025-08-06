# Weighted Self-Avoiding Walk and Trail Analyzer

This project provides a Python-based framework for analyzing weighted Self-Avoiding Walks (SAWs) and Self-Avoiding Trails (SATs) on various lattices. It uses the numerical method developed in the accompanying paper **[arXiv:2508.01993](https://arxiv.org/abs/2508.01993)** to derive rigorous upper bounds on the corresponding weighted connective constants.

The program is designed to be extensible, allowing for the easy addition of new lattices and weighting schemes.

-----

## Features

  - **Extensibility**: New lattices (e.g., BCC, FCC) and weighting schemes can be added by creating a new Python file in the `lattices/` directory that adheres to the base `Lattice` interface. The main program automatically discovers and registers them.
  - **Multiple Object Types**: Supports both **Self-Avoiding Walks** (which cannot revisit a vertex) and **Self-Avoiding Trails** (which cannot reuse an edge).
  - **Symbolic Computation**: Leverages the `sympy` library to construct the G-matrix symbolically.
  - **Result Caching**: Automatically caches the computed G-matrices to a `cache/` directory.

-----

## Code Structure

```
.
├── main.py                      # Main executable script with the analysis logic and CLI.
├── plot_config.py               # Configuration for plot insets and views.
├── lattices/
│   ├── __init__.py              # Package initializer with dynamic lattice discovery.
│   ├── base.py                  # Defines the abstract Lattice class interface.
│   ├── square.py                # Implementation for the 2D square lattice.
│   ├── triangular.py            # Implementation for the 2D triangular lattice.
│   ├── triangular_reduced.py    # Implementation for the 2D triangular lattice with a reduced weighting scheme.
│   ├── simple-cubic.py          # Implementation for the 3D simple-cubic lattice.
│   └── simple-cubic_reduced.py  # Implementation for the 3D simple-cubic lattice with a reduced weighting scheme.
└── cache/                       # (Auto-generated) Stores cached G-matrices.
```

-----

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required Python packages. A `virtualenv` is recommended.

    ```bash
    pip install sympy matplotlib numpy scikit-image tqdm
    ```

3.  Ensure you have a working LaTeX distribution (like MiKTeX, TeX Live, or MacTeX) installed for `matplotlib` to render plots with TeX fonts.

-----

## Usage

The main script is `main.py`. You can run it from the command line to perform different actions.

### Basic Syntax

```bash
python main.py --lattice <name> --mode <type> --pairs <m1,n1> [<m2,n2> ...] --action <action>
```

  - `--lattice`: The name of the lattice to use (e.g., `square`, `simple-cubic`, `triangular`).
  - `--mode`: The type of self-avoiding object (`walk` or `trail`).
  - `--pairs`: One or more space-separated `m,n` pairs. **Note: `m` must be less than `n`**.
  - `--action`: The desired output:
      - `plot`: Generate and save a `.pgf` plot file.
      - `matrix`: Print the computed G-matrix to the console in a human-readable format.
      - `latex`: Print the G-matrix in LaTeX format.

### Examples

**1. Generate a 2D Contour Plot for SAWs on the Square Lattice**

This command computes the G-matrices for (m=2, n=4) and (m=3, n=5) and plots the `lambda_1 = 1` contours. The output will be saved as `square_walk.pgf`.

```bash
python main.py --lattice square --mode walk --pairs 2,4 3,5 --action plot
```

**2. Generate 3D Surface Plots for SATs on the Simple-Cubic Lattice**

This command analyzes self-avoiding trails for the pair (m=2, n=3) on the simple-cubic lattice. It will produce two files: `simple-cubic_trail_full.pgf` (the full surface) and `simple-cubic_trail_local.pgf` (a zoomed-in view near the isotropic line).

```bash
python main.py --lattice simple-cubic --mode trail --pairs 2,3 --action plot
```

**3. Print the Symbolic G-Matrix**

This command computes the G-matrix for (m=2, n=3) on the triangular lattice and prints it to the terminal.

```bash
python main.py --lattice triangular --mode walk --pairs 2,3 --action matrix
```

**4. Get the LaTeX Code for a G-Matrix**

This command prints the G-matrix in a format ready for a TeX document:

```bash
python main.py --lattice square --mode trail --pairs 2,4 --action latex
```

---

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@online{he2025upperboundsconnectiveconstant,
      title={Upper bounds for the connective constant of weighted self-avoiding walks}, 
      author={Qidong He},
      year={2025},
      eprint={2508.01993},
      archivePrefix={arXiv},
      primaryClass={math.PR}
}
```
