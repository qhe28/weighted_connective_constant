import sympy
import matplotlib
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

def is_primitive(matrix: np.ndarray) -> bool:
    """
    Checks if a non-negative square matrix is primitive.
    A matrix A is primitive iff A^(t^2 - 2t + 2) is positive, where t is the dimension of the matrix.
    """
    if not np.all(matrix >= 0):
        print("Warning: Matrix contains negative entries, cannot check primitivity.")
        return False
    else:
        t = matrix.shape[0]
        if t == 0:
            return True
        elif t == 1:
            return matrix[0,0] > 0
        else:
            try:
                power = t**2 - 2*t + 2
                powered_matrix = np.linalg.matrix_power(matrix, power)
                return np.all(powered_matrix > 0)
            except np.linalg.LinAlgError:
                print("Warning: Could not compute matrix power to check primitivity.")
                return False


# --- Implementation of the method ---
class Analyzer:
    """
    Implements the method for any lattice defined by a Lattice object.
    """
    def __init__(self, m: int, n: int, lattice: lattices.base.Lattice, mode: str = 'walk'):
        if m < 0 or n <= m:
            raise ValueError(f"m must be non-negative and less than n, but got m={m}, n={n}.")
        else:
            self.m = m
            self.n = n
            self.lattice = lattice
            self.mode = mode
            self._generated_walks = {}
        
    def _generate_walks(self, length: int, starting_point: tuple):
        if (length, starting_point) in self._generated_walks:
            return
        else:
            initial_steps = self.lattice.get_neighbor_vectors(starting_point)
            initial_walks = [[starting_point, tuple(starting_point[i] + step[i] for i in range(self.lattice.dim))] 
                                for step in initial_steps]
            generated_walks = {1: initial_walks}
            
            for l in range(2, length + 1):
                prev_walks = generated_walks[l - 1]
                new_walks = []
                
                for walk in tqdm(prev_walks, desc=f"Generating L={l} {self.mode}s from {starting_point}"):
                    last_vertex = walk[-1]
                    visited_vertices = set(walk) if self.mode == 'walk' else set()
                    visited_edges = {tuple(sorted((walk[i], walk[i+1]))) for i in range(len(walk) - 1)} if self.mode == 'trail' else set()
                    
                    for step in self.lattice.get_neighbor_vectors(last_vertex):     # Attempts extension by one step
                        next_vertex = tuple(last_vertex[i] + step[i] for i in range(self.lattice.dim))
                        is_valid = not ((self.mode == 'walk' and next_vertex in visited_vertices) or \
                                        (self.mode == 'trail' and tuple(sorted((last_vertex, next_vertex))) in visited_edges))
                        if is_valid:
                            new_walks.append(walk + [next_vertex])
                            
                generated_walks[l] = new_walks
            
            for l, walks in generated_walks.items():
                key = (l, starting_point)
                if key not in self._generated_walks:
                    self._generated_walks[key] = []
                self._generated_walks[key].extend(walks)

    def get_equivalence_classes(self):
        """
        Generates m-step self-avoiding walks starting from each vertex class representative,
        and then partitions the combined set into equivalence classes.
        """
        m_step_walks = []
        
        if self.m == 0:
            print(f"Generating 0-step walks (single vertices) for {len(self.lattice.vertex_reps)} vertex class(es)...")
            for vertex_rep in self.lattice.vertex_reps:
                m_step_walks.append([vertex_rep])
        else:
            print(f"Generating m-step walks from {len(self.lattice.vertex_reps)} vertex class(es)...")
            for vertex_rep in self.lattice.vertex_reps:
                self._generate_walks(self.m, starting_point=vertex_rep)
                generated_walks = self._generated_walks.get((self.m, vertex_rep), [])
                m_step_walks.extend(generated_walks)
                print(f"  ...found {len(generated_walks)} walks from representative {vertex_rep}.")

        if not m_step_walks:
            raise RuntimeError(f"Failed to generate any walks for m={self.m}.")
        else:
            print(f"Total m-step walks to classify: {len(m_step_walks)}")
            
            equiv_classes = defaultdict(list)
            for walk in tqdm(m_step_walks, desc=f"Finding equivalence classes for m={self.m}"):
                canonical_form = self.lattice.get_canonical_form(tuple(map(tuple, walk)))
                equiv_classes[canonical_form].append(walk)

            equiv_class_reps = sorted(equiv_classes.keys())
            equiv_class_map = {rep: i for i, rep in enumerate(equiv_class_reps)}
            
            print(f"Found {len(equiv_class_reps)} equivalence classes for m={self.m} on the {self.lattice.name} lattice.")
            return equiv_class_reps, equiv_class_map

    def _build_g_row(self, visited_vertices, visited_edges, current_weight, remaining_steps, row_vector, equiv_class_map):
        if remaining_steps == 0:
            tail = tuple(map(tuple, visited_vertices[-self.m-1:]))
            canonical_tail = self.lattice.get_canonical_form(tail)
            s = equiv_class_map[canonical_tail]
            row_vector[s] += current_weight
        else:
            last_vertex = visited_vertices[-1]
            for step in self.lattice.get_neighbor_vectors(last_vertex):
                next_vertex = tuple(last_vertex[i] + step[i] for i in range(self.lattice.dim))
                is_valid = True
                if self.mode == 'walk' and next_vertex in set(visited_vertices):
                    is_valid = False
                elif self.mode == 'trail' and tuple(sorted((last_vertex, next_vertex))) in visited_edges:
                    is_valid = False

                if is_valid:
                    new_visited_edges = visited_edges.copy()
                    if self.mode == 'trail':
                        new_visited_edges.add(tuple(sorted((last_vertex, next_vertex))))
                    
                    step_weight = self.lattice.get_step_weight(last_vertex, next_vertex)
                    self._build_g_row(
                        visited_vertices + [next_vertex], new_visited_edges, current_weight * step_weight, 
                        remaining_steps - 1, row_vector, equiv_class_map
                    )
                
    def compute_g(self):
        """
        Computes G(m, n), using a cache to load/save results.
        """
        cache_dir = "cache"
        cache_filename = f"G_{self.m}_{self.n}_{self.lattice.name}_{self.mode}.pkl"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading G({self.m}, {self.n}) from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                return pickle.load(f)
        else:
            equiv_class_reps, equiv_class_map = self.get_equivalence_classes()
            t = len(equiv_class_reps)
            g = sympy.zeros(t, t)

            print(f"Constructing {t}x{t} G({self.m}, {self.n})...")
            
            for r, m_step_walk in enumerate(tqdm(equiv_class_reps, desc=f"Building G({self.m}, {self.n})")):
                initial_vertices = list(map(tuple, m_step_walk))
                row_vector = [0] * t
                initial_edges = {tuple(sorted((initial_vertices[i], initial_vertices[i+1]))) for i in range(len(initial_vertices) - 1)}
                            
                self._build_g_row(
                    visited_vertices=initial_vertices, visited_edges=initial_edges, current_weight=1,
                    remaining_steps=self.n - self.m, row_vector=row_vector, equiv_class_map=equiv_class_map
                )
                
                g[r] = row_vector
            
            print(f"Saving G({self.m}, {self.n}) to cache: {cache_filepath}")
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(g, f)
                
            return g


# --- Plotting functions ---
def create_2d_plot_figure(results_list, plot_symbols, inset_config=None):
    """
    Creates a figure and axes for a 2D contour plot.
    Does not save or show the plot, just returns the figure object.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel(f'${sympy.latex(plot_symbols[0])}$')
    ax.set_ylabel(f'${sympy.latex(plot_symbols[1])}$')
    grid_res = 100
    x_vals = np.linspace(0.0, 1.0, grid_res)
    y_vals = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.05, -0.05, 1, 1),
                          bbox_transform=ax.transAxes)

    for i, (g, m, n, mode, _) in enumerate(results_list):
        print(f"Processing (m={m}, n={n}) for {mode}s...")
        g_func = sympy.lambdify(plot_symbols, g, 'numpy')
        Z = np.zeros_like(X)

        for row in tqdm(range(grid_res), desc=f"Calculating grid for (m={m}, n={n})"):
            for col in range(grid_res):
                x_val, y_val = X[row, col], Y[row, col]
                try:
                    numeric_matrix = g_func(x_val, y_val).astype(np.float64)
                    eigenvalues = np.linalg.eigvals(numeric_matrix)
                    lambda_1 = np.max(np.real(eigenvalues))
                    Z[row, col] = lambda_1
                except Exception:
                    Z[row, col] = np.nan

        ax.contour(X, Y, Z, levels=[1.0], colors=[colors[i]])
        ax_inset.contour(X, Y, Z, levels=[1.0], colors=[colors[i]])

    if inset_config:
        ax_inset.set_xlim(inset_config['xlim'])
        ax_inset.set_ylim(inset_config['ylim'])
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True)

    handles = [plt.Line2D([0], [0], color=colors[i], label=fr'$(m={r[1]}, n={r[2]})$') for i, r in enumerate(results_list)]
    
    # Uncomment the following lines to add isotropic line
    # ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    # ax_inset.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    # handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', label=r'Isotropic ($y=x$)'))
    
    ax.legend(handles=handles, loc='upper left')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def create_3d_plot_figures(results_list, plot_symbols, local_config=None):
    """
    Creates figures for 3D surface plots (full and local views).
    Returns a tuple of figure walks: (fig_full, fig_local).
    """
    fig_full = plt.figure(figsize=(10, 8))
    ax_full = fig_full.add_subplot(111, projection='3d')
    ax_full.set_xlabel(f'${sympy.latex(plot_symbols[0])}$')
    ax_full.set_ylabel(f'${sympy.latex(plot_symbols[1])}$')
    ax_full.set_zlabel(f'${sympy.latex(plot_symbols[2])}$')
    grid_res_full = 20
    ax_full.set_xlim(0, 1); ax_full.set_ylim(0, 1); ax_full.set_zlim(0, 1)
    vals_full = np.linspace(0, 1, grid_res_full)
    
    fig_local = plt.figure(figsize=(8, 8))
    ax_local = fig_local.add_subplot(111, projection='3d')
    ax_local.set_xlabel(f'${sympy.latex(plot_symbols[0])}$')
    ax_local.set_ylabel(f'${sympy.latex(plot_symbols[1])}$')
    ax_local.set_zlabel(f'${sympy.latex(plot_symbols[2])}$')
    grid_res_local = 20
    if local_config:
        ax_local.set_xlim(local_config['xlim']); ax_local.set_ylim(local_config['ylim']); ax_local.set_zlim(local_config['zlim'])
        x_vals_local = np.linspace(local_config['xlim'][0], local_config['xlim'][1], grid_res_local)
        y_vals_local = np.linspace(local_config['ylim'][0], local_config['ylim'][1], grid_res_local)
        z_vals_local = np.linspace(local_config['zlim'][0], local_config['zlim'][1], grid_res_local)
    
    all_plot_data = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    for i, (g, m, n, mode, _) in enumerate(results_list):
        print(f"Processing (m={m}, n={n}) for {mode}s...")
        g_func = sympy.lambdify(plot_symbols, g, 'numpy')
        
        lambda_field_full = np.zeros((grid_res_full, grid_res_full, grid_res_full))
        for ix in tqdm(range(grid_res_full), desc=f"Calculating full grid for (m={m}, n={n})"):
            for iy in range(grid_res_full):
                for iz in range(grid_res_full):
                    try:
                        mat = g_func(vals_full[ix], vals_full[iy], vals_full[iz])
                        lambda_field_full[ix, iy, iz] = np.max(np.real(np.linalg.eigvals(mat)))
                    except Exception: lambda_field_full[ix, iy, iz] = np.nan
        
        lambda_field_local = np.zeros((grid_res_local, grid_res_local, grid_res_local))
        if local_config:
            for ix in tqdm(range(grid_res_local), desc=f"Calculating local grid for (m={m}, n={n})"):
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

    for data in all_plot_data:
        try:
            verts, faces, _, _ = measure.marching_cubes(data['field_full'], level=1.0, spacing=(1.0/grid_res_full, 1.0/grid_res_full, 1.0/grid_res_full))
            ax_full.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=data['color'], alpha=1.0, linewidth=0.0)
        except (ValueError, RuntimeError) as e:
            print(f"Could not generate main surface for (m={data['m']}, n={data['n']}): {e}")

    if local_config:
        for data in all_plot_data:
            try:
                spacing_local = ((local_config['xlim'][1] - local_config['xlim'][0]) / grid_res_local,
                                (local_config['ylim'][1] - local_config['ylim'][0]) / grid_res_local,
                                (local_config['zlim'][1] - local_config['zlim'][0]) / grid_res_local)
                verts, faces, _, _ = measure.marching_cubes(data['field_local'], level=1.0, spacing=spacing_local)
                verts += [local_config['xlim'][0], local_config['ylim'][0], local_config['zlim'][0]]
                ax_local.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=data['color'], alpha=0.5, linewidth=0.0)
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Could not generate local surface for (m={data['m']}, n={data['n']}): {e}")

    handles = [plt.Rectangle((0,0),1,1, color=d['color']) for d in all_plot_data]
    labels = [f"$(m={d['m']}, n={d['n']})$" for d in all_plot_data]
    ax_full.legend(handles=handles, loc='upper left', labels=labels); ax_local.legend(handles=handles, loc='upper left', labels=labels)
    ax_full.grid(True); ax_local.grid(True)
    ax_full.view_init(elev=30, azim=45); ax_local.view_init(elev=15, azim=-35)
    for axis in [ax_full.xaxis, ax_full.yaxis, ax_full.zaxis]: axis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    for axis in [ax_local.xaxis, ax_local.yaxis, ax_local.zaxis]: axis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    
    return fig_full, fig_local

def parse_pairs(value):
    """Custom type for argparse to parse m,n pairs."""
    try:
        m, n = map(int, value.split(','))
        if n <= m:
            raise argparse.ArgumentTypeError(f"m must be < n in '{value}'")
        return m, n
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: '{value}'. Use 'm,n'.")


if __name__ == '__main__':
    available_lattice_names = list(lattices.get_available_lattices().keys())
    
    parser = argparse.ArgumentParser(description="Analyzer for anisotropic SAWs/SATs.")
    parser.add_argument('--lattice', type=str, choices=available_lattice_names, required=True, help="Lattice to analyze.")
    parser.add_argument('--mode', type=str, choices=['walk', 'trail'], required=True, help="Type of self-avoiding object.")
    parser.add_argument('--pairs', type=parse_pairs, nargs='+', required=True, help="One or more m,n pairs (e.g., 0,2 1,3).")
    parser.add_argument('--action', type=str, choices=['pgf', 'plot', 'matrix', 'latex'], required=True, help="Action: 'pgf' to save file, 'plot' to display, 'matrix'/'latex' to print.")
    
    args = parser.parse_args()

    selected_lattice = lattices.get_lattice(args.lattice)

    results_for_analysis = []
    
    try:
        for m_val, n_val in args.pairs:
            print(f"\n--- Starting: {args.lattice.upper()} (m={m_val}, n={n_val}), MODE={args.mode} ---")
            analyzer = Analyzer(m=m_val, n=n_val, lattice=selected_lattice, mode=args.mode)
            g = analyzer.compute_g()
            
            plot_symbols = sorted(list(g.free_symbols), key=lambda s: s.name)
            
            g_binary = np.array(g.applyfunc(lambda element: 0 if element.is_zero else 1))   # Check for primitivity using a binary matrix
            if is_primitive(g_binary):
                print(f"SUCCESS: G({m_val}, {n_val}) is primitive.")
            else:
                print(f"WARNING: G({m_val}, {n_val}) may NOT be primitive.")
            
            results_for_analysis.append((g, m_val, n_val, args.mode, plot_symbols))

        if results_for_analysis:
            first_result_symbols = results_for_analysis[0][4]
            num_plot_vars = len(first_result_symbols)

            # --- Action Dispatch ---
            if args.action == 'pgf' or args.action == 'plot':
                is_saving_file = (args.action == 'pgf')
                
                if is_saving_file:
                    matplotlib.use('pgf')
                    matplotlib.rcParams.update({
                        "pgf.texsystem": "pdflatex",
                        'font.family': 'serif',
                        'text.usetex': True,
                        'pgf.rcfonts': False,
                    })

                if num_plot_vars == 2:
                    fig = create_2d_plot_figure(results_for_analysis, first_result_symbols, plot_config.get_config(args.lattice, args.mode))
                    fig.canvas.manager.set_window_title(f"{args.lattice}_{args.mode}")
                    if is_saving_file:
                        output_filename = f"{args.lattice}_{args.mode}.pgf"
                        print(f"Saving plot to {output_filename}...")
                        fig.savefig(output_filename, bbox_inches='tight')
                        print("Plot saved.")
                    else:
                        plt.show()

                elif num_plot_vars == 3:
                    fig_full, fig_local = create_3d_plot_figures(results_for_analysis, first_result_symbols, plot_config.get_config(args.lattice, args.mode))
                    fig_full.canvas.manager.set_window_title(f"{args.lattice}_{args.mode}_full")
                    fig_local.canvas.manager.set_window_title(f"{args.lattice}_{args.mode}_local")
                    if is_saving_file:
                        output_full = f"{args.lattice}_{args.mode}_full.pgf"
                        output_local = f"{args.lattice}_{args.mode}_local.pgf"
                        print(f"Saving full plot to {output_full}...")
                        fig_full.savefig(output_full, bbox_inches='tight')
                        print(f"Saving local plot to {output_local}...")
                        fig_local.savefig(output_local, bbox_inches='tight')
                        print("PGF files saved successfully.")
                    else:
                        plt.show()
                else:
                    print(f"Plotting not supported for {num_plot_vars} variables. Please use action 'matrix' or 'latex'.")
            
            elif args.action == 'matrix':
                for g, m, n, mode, _ in results_for_analysis:
                    print(f"\nG({m}, {n}) for {mode}s on the {selected_lattice.name} lattice:")
                    print(sympy.pretty(g, use_unicode=True))
            
            elif args.action == 'latex':
                for g, m, n, mode, _ in results_for_analysis:
                    print(f"\nG({m}, {n}) for {mode}s on the {selected_lattice.name} lattice:")
                    print(latex(g))

    except (ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}")
