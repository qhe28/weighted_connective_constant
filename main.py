import sympy
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from tqdm import tqdm
import os
import pickle
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage import measure
from matplotlib.ticker import MaxNLocator
from sympy import latex

import plot_config
import lattices

# Configure Matplotlib for PGF output and LaTeX fonts
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# --- Analyzer ---
class AnisotropicSAW:
    """
    Implements the algorithm for any lattice defined by a Lattice object.
    """
    def __init__(self, m: int, n: int, lattice: lattices.base.Lattice, mode: str = 'walk'):
        if m >= n:
            raise ValueError(f"m must be less than n, but got m={m}, n={n}.")
        self.m = m
        self.n = n
        self.lattice = lattice
        self.mode = mode
        # This dictionary will store generated objects for the lifetime of the instance.
        self._generated_objects = {}
        
    def _generate_objects_iteratively(self, max_len: int):
        """
        Generates objects (walks or trails) up to a specified length,
        storing them in the instance's _generated_objects dictionary.
        """
        # Start with objects of length 1.
        start_node = tuple([0] * self.lattice.dim)
        current_objects = [[start_node, neighbor] for neighbor in self.lattice.neighbors]
        self._generated_objects[1] = current_objects
        print(f"Generated {len(current_objects)} {self.mode}s of length 1.")

        # Iteratively generate objects for lengths 2 up to max_len.
        for length in range(2, max_len + 1):
            prev_objects = self._generated_objects[length - 1]
            new_objects = []
            
            for path in tqdm(prev_objects, desc=f"Generating length {length} {self.mode}s"):
                last_node = path[-1]
                
                # Determine visited nodes or edges based on the mode.
                visited_nodes = set(path) if self.mode == 'walk' else set()
                visited_edges = {tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1)} if self.mode == 'trail' else set()

                for step in self.lattice.neighbors:
                    next_node = tuple(last_node[i] + step[i] for i in range(self.lattice.dim))
                    
                    is_valid = True
                    if self.mode == 'walk' and next_node in visited_nodes:
                        is_valid = False
                    elif self.mode == 'trail' and tuple(sorted((last_node, next_node))) in visited_edges:
                        is_valid = False
                    
                    if is_valid:
                        new_objects.append(path + [next_node])
            
            self._generated_objects[length] = new_objects
            print(f"Generated {len(new_objects)} {self.mode}s of length {length}.")

    def get_equivalence_classes(self):
        """
        Generates objects of length m and groups them into equivalence classes
        based on their canonical form.
        """
        # Ensure objects of length m are generated.
        self._generate_objects_iteratively(self.m)
        all_m_objects = self._generated_objects.get(self.m, [])
        if not all_m_objects:
            raise RuntimeError(f"Failed to generate objects for key {self.m}.")

        classes = defaultdict(list)
        for walk in tqdm(all_m_objects, desc=f"Finding equivalence classes for m={self.m}"):
            # The walk must be converted to a tuple of tuples to be hashable.
            canonical_form = self.lattice.get_canonical_form(tuple(map(tuple, walk)))
            classes[canonical_form].append(walk)

        class_reps = sorted(classes.keys())
        class_map = {rep: i for i, rep in enumerate(class_reps)}
        
        print(f"Found {len(class_reps)} equivalence classes for m={self.m} on the {self.lattice.name} lattice.")
        return class_reps, class_map

    def _calculate_weight(self, walk):
        """Calculates the symbolic weight of a given walk."""
        weight = 1
        for i in range(len(walk) - 1):
            weight *= self.lattice.get_step_weight(walk[i], walk[i+1])
        return weight

    def _build_g_recursively(self, path, edges, current_weight, remaining_steps, row_vector, class_map):
        """Recursively explores paths to build a row of the G matrix."""
        if remaining_steps == 0:
            # Tail must be converted to tuple of tuples for hashing and finding its canonical form.
            tail = tuple(map(tuple, path[-self.m-1:]))
            canonical_tail = self.lattice.get_canonical_form(tail)
            if canonical_tail in class_map:
                s = class_map[canonical_tail]
                row_vector[s] += current_weight
            return

        last_node = path[-1]
        for step in self.lattice.neighbors:
            next_node = tuple(last_node[i] + step[i] for i in range(self.lattice.dim))
            
            is_valid = True
            if self.mode == 'walk' and next_node in path:
                is_valid = False
            elif self.mode == 'trail' and tuple(sorted((last_node, next_node))) in edges:
                is_valid = False

            if is_valid:
                new_edges = edges.copy()
                if self.mode == 'trail':
                    new_edges.add(tuple(sorted((last_node, next_node))))
                
                step_weight = self.lattice.get_step_weight(last_node, next_node)
                self._build_g_recursively(
                    path + [next_node], new_edges, current_weight * step_weight, 
                    remaining_steps - 1, row_vector, class_map
                )

    def compute_g(self):
        """
        Computes the G matrix, using a cache to load/save results.
        """
        # --- Caching Logic Start ---
        cache_dir = "cache"
        cache_filename = f"g_{self.lattice.name}_{self.mode}_{self.m}_{self.n}.pkl"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading G matrix from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                return pickle.load(f)
        # --- Caching Logic End ---

        class_reps, class_map = self.get_equivalence_classes()
        t = len(class_reps)
        g = sympy.zeros(t, t)

        print(f"Constructing {t}x{t} G({self.m}, {self.n}) matrix...")
        
        for r, start_walk_tuple in enumerate(tqdm(class_reps, desc="Building G")):
            start_walk = list(map(tuple, start_walk_tuple)) # Convert back to list of lists
            row_vector = [0] * t
            initial_edges = {tuple(sorted((start_walk[i], start_walk[i+1]))) for i in range(len(start_walk) - 1)}
            
            initial_weight = self._calculate_weight(start_walk)
            
            self._build_g_recursively(
                path=start_walk, edges=initial_edges, current_weight=initial_weight, 
                remaining_steps=self.n - self.m, row_vector=row_vector, class_map=class_map
            )
            
            for s in range(t):
                if initial_weight != 0:
                    g[r, s] = sympy.expand(row_vector[s] / initial_weight)
                else:
                    g[r, s] = 0
        
        # --- Caching Logic Start ---
        print(f"Saving G matrix to cache: {cache_filepath}")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_filepath, 'wb') as f:
            pickle.dump(g, f)
        # --- Caching Logic End ---

        return g


# --- Plotting and Main Execution ---
def plot_2d(results_list, output_filename, inset_config=None):
    """
    Numerically computes and plots the lambda_1(G)=1 contour for a list of G matrices,
    saving the result to a PGF file. Includes an inset plot to show detail.

    Args:
        results_list (list): A list of tuples, where each tuple is
                             (g_matrix, m, n, mode).
        output_filename (str): The path to save the PGF file.
    """
    print(f"Starting analysis for contour plotting. Will save to {output_filename}")
    
    fig, ax = plt.subplots(figsize=(5, 5)) # Adjusted for typical TeX document width
    ax.set_xlabel('$x$ (horizontal weight)')
    ax.set_ylabel('$y$ (vertical weight)')
    grid_res = 100
    x_vals = np.linspace(0.0, 1.0, grid_res)
    y_vals = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    # Create the inset axes
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.05, -0.05, 1, 1),
                          bbox_transform=ax.transAxes)

    for i, (g, m, n, mode, symbols) in enumerate(results_list):
        print(f"Processing (m={m}, n={n}) for {mode}s...")
        g_func = sympy.lambdify(selected_lattice.symbols, g, 'numpy')
        Z = np.zeros_like(X)

        for row in tqdm(range(grid_res), desc=f"Calculating grid for ({m},{n})"):
            for col in range(grid_res):
                x_val, y_val = X[row, col], Y[row, col]
                try:
                    numeric_matrix = g_func(x_val, y_val).astype(np.float64)
                    eigenvalues = np.linalg.eigvals(numeric_matrix)
                    lambda_1 = np.max(np.real(eigenvalues))
                    Z[row, col] = lambda_1
                except Exception:
                    Z[row, col] = np.nan

        # Plot on the main axes and inset axes
        ax.contour(X, Y, Z, levels=[1.0], colors=[colors[i]])
        ax_inset.contour(X, Y, Z, levels=[1.0], colors=[colors[i]])

    # Configure the inset plot's view
    if inset_config:
        ax_inset.set_xlim(inset_config['xlim'])
        ax_inset.set_ylim(inset_config['ylim'])
        ax_inset.set_xticks(inset_config['ticks'])
        ax_inset.set_yticks(inset_config['ticks'])
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True)

    # Add the y=x line to both plots
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax_inset.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

    # Create legend for the main plot
    # Use raw strings for LaTeX compatibility
    handles = [plt.Line2D([0], [0], color=colors[i], label=fr'$(m={r[1]}, n={r[2]})$') for i, r in enumerate(results_list)]
    handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', label=r'Isotropic ($y=x$)'))
    ax.legend(handles=handles, loc='upper left')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save the figure
    print(f"Saving plot to {output_filename}...")
    plt.savefig(output_filename, bbox_inches='tight')
    print("Plot saved.")


def plot_3d(results_list, output_full_filename, output_local_filename, local_config=None):
    """
    Creates and saves two 3D surface plots as PGF files.
    """
    print("Starting 3D analysis for surface plotting.")
    
    fig_full = plt.figure(figsize=(10, 8))
    ax_full = fig_full.add_subplot(111, projection='3d')
    ax_full.set_xlabel('$x$')
    ax_full.set_ylabel('$y$')
    ax_full.set_zlabel('$z$')
    grid_res_full = 20
    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(0, 1)
    ax_full.set_zlim(0, 1)
    vals_full = np.linspace(0, 1, grid_res_full)
    
    fig_local = plt.figure(figsize=(8, 8))
    ax_local = fig_local.add_subplot(111, projection='3d')
    ax_local.set_xlabel('$x$')
    ax_local.set_ylabel('$y$')
    ax_local.set_zlabel('$z$')
    grid_res_local = 20
    if local_config:
        ax_local.set_xlim(local_config['xlim'])
        ax_local.set_ylim(local_config['ylim'])
        ax_local.set_zlim(local_config['zlim'])
        x_vals_local = np.linspace(local_config['xlim'][0], local_config['xlim'][1], grid_res_local)
        y_vals_local = np.linspace(local_config['ylim'][0], local_config['ylim'][1], grid_res_local)
        z_vals_local = np.linspace(local_config['zlim'][0], local_config['zlim'][1], grid_res_local)
    
    all_plot_data = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    for i, (g, m, n, mode) in enumerate(results_list):
        print(f"Processing (m={m}, n={n}) for {mode}s...")
        g_func = sympy.lambdify(selected_lattice.symbols, g, 'numpy')
        
        lambda_field_full = np.zeros((grid_res_full, grid_res_full, grid_res_full))
        for ix in tqdm(range(grid_res_full), desc=f"Calculating full grid for ({m},{n})"):
            for iy in range(grid_res_full):
                for iz in range(grid_res_full):
                    try:
                        mat = g_func(vals_full[ix], vals_full[iy], vals_full[iz])
                        lambda_field_full[ix, iy, iz] = np.max(np.real(np.linalg.eigvals(mat)))
                    except Exception: lambda_field_full[ix, iy, iz] = np.nan
        
        lambda_field_local = np.zeros((grid_res_local, grid_res_local, grid_res_local))
        for ix in tqdm(range(grid_res_local), desc=f"Calculating local grid for ({m},{n})"):
            for iy in range(grid_res_local):
                for iz in range(grid_res_local):
                    try:
                        mat = g_func(x_vals_local[ix], y_vals_local[iy], z_vals_local[iz])
                        lambda_field_local[ix, iy, iz] = np.max(np.real(np.linalg.eigvals(mat)))
                    except Exception: lambda_field_local[ix, iy, iz] = np.nan
        
        all_plot_data.append({
            'm': m, 'n': n, 'color': colors[i],
            'field_full': lambda_field_full, 'field_local': lambda_field_local
        })

    # Generate the full plot
    for data in all_plot_data:
        try:
            verts, faces, _, _ = measure.marching_cubes(data['main_field'], level=1.0, spacing=(1.0/grid_res_full, 1.0/grid_res_full, 1.0/grid_res_full))
            ax_full.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=data['color'], alpha=1.0, linewidth=0.0)
        except (ValueError, RuntimeError) as e:
            print(f"Could not generate main surface for (m={data['m']}, n={data['n']}): {e}")
    ax_full.view_init(elev=30, azim=45)
    handles = [plt.Rectangle((0,0),1,1, color=d['color']) for d in all_plot_data]
    labels = [f"$(m={d['m']}, n={d['n']})$" for d in all_plot_data]
    ax_full.legend(handles=handles, loc='upper left', labels=labels)
    ax_full.grid(True)
    for axis in [ax_full.xaxis, ax_full.yaxis, ax_full.zaxis]: 
        axis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # Generate the local plot
    for data in all_plot_data:
        try:
            spacing_local = ((local_config['xlim'][1] - local_config['xlim'][0]) / grid_res_local,
                             (local_config['ylim'][1] - local_config['ylim'][0]) / grid_res_local,
                             (local_config['zlim'][1] - local_config['zlim'][0]) / grid_res_local)
            verts, faces, _, _ = measure.marching_cubes(data['field_local'], level=1.0, spacing=spacing_local)
            verts += [local_config['xlim'][0], local_config['ylim'][0], local_config['zlim'][0]]
            ax_local.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=data['color'], alpha=0.5, linewidth=0.0)
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"Could not generate inset surface for (m={data['m']}, n={data['n']}): {e}")
    ax_local.view_init(elev=15, azim=-35)
    ax_local.legend(handles=handles, loc='upper left', labels=labels)
    ax_local.grid(True)
    for axis in [ax_local.xaxis, ax_local.yaxis, ax_local.zaxis]: 
        axis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    print(f"Saving full plot to {output_full_filename}...")
    fig_full.savefig(output_full_filename, bbox_inches='tight')
    print(f"Saving local plot to {output_local_filename}...")
    fig_local.savefig(output_local_filename, bbox_inches='tight')
    plt.close(fig_full); plt.close(fig_local)
    print("PGF files saved successfully.")


def parse_pairs(value):
    """Custom type for argparse to parse m,n pairs."""
    try:
        m, n = map(int, value.split(','))
        if m >= n:
            raise argparse.ArgumentTypeError(f"m must be < n in '{value}'")
        return m, n
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: '{value}'. Use 'm,n'.")


if __name__ == '__main__':
    # Get the list of available lattices dynamically for the help message.
    available_lattice_names = list(lattices.get_available_lattices().keys())
    
    parser = argparse.ArgumentParser(description="Analyzer for anisotropic SAWs/SATs.")
    parser.add_argument('--lattice', type=str, choices=available_lattice_names, required=True, help="Lattice to analyze.")
    parser.add_argument('--mode', type=str, choices=['walk', 'trail'], required=True, help="Type of self-avoiding object.")
    parser.add_argument('--pairs', type=parse_pairs, nargs='+', required=True, help="One or more m,n pairs (e.g., 2,4 3,5).")
    parser.add_argument('--action', type=str, choices=['plot', 'matrix', 'latex'], required=True, help="Generate a .pgf file or print a list of G matrices.")
    
    args = parser.parse_args()

    # Get the selected lattice object from the package.
    selected_lattice = lattices.get_lattice(args.lattice)

    results_for_analysis = []
    
    try:
        for m_val, n_val in args.pairs:
            print(f"--- Starting: {args.lattice.upper()} (m={m_val}, n={n_val}), MODE={args.mode} ---")
            analyzer = AnisotropicSAW(m=m_val, n=n_val, lattice=selected_lattice, mode=args.mode)
            g = analyzer.compute_g()
            # print(f"Computed G({m_val}, {n_val}):\n{g}")
            results_for_analysis.append((g, m_val, n_val, args.mode, selected_lattice.symbols))

        if results_for_analysis:
            if args.action == 'plot':
                # --- Plotting Dispatch ---
                if selected_lattice.name == 'square':
                    output_filename = f"{args.lattice}_{args.mode}.pgf"
                    inset_cfg = plot_config.get_config(args.lattice, args.mode)
                    plot_2d(results_for_analysis, output_filename, inset_config=inset_cfg)
                elif selected_lattice.dim in ('simple-cubic', 'triangular'):
                    output_full_filename = f"{args.lattice}_{args.mode}_full.pgf"
                    output_local_filename = f"{args.lattice}_{args.mode}_local.pgf"
                    local_cfg = plot_config.get_config(args.lattice, args.mode)
                    plot_3d(results_for_analysis, output_full_filename, output_local_filename, local_config=local_cfg)
            elif args.action == 'matrix':
                # Print the G matrices in a readable format.
                for g, m, n, mode, symbols in results_for_analysis:
                    print(f"G({m}, {n}) for {mode}s on the {selected_lattice.name} lattice:")
                    print(sympy.pretty(g, use_unicode=True))
                    # print("\n" + "="*40 + "\n")
            elif args.action == 'matrix_latex':
                # Print the G matrices in LaTeX.
                for g, m, n, mode, symbols in results_for_analysis:
                    print(f"G({m}, {n}) for {mode}s on the {selected_lattice.name} lattice:")
                    print(latex(g))
                    # print("\n" + "="*40 + "\n")
            

    except (ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}")
