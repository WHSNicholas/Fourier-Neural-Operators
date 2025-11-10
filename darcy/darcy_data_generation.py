# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                         2D Darcy Equation - Data Generation                                          #
# -------------------------------------------------------------------------------------------------------------------- #


# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dependencies
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
from utils.data import sample_field_2d, gather_function_data

# Parallel Computing
comm = MPI.COMM_WORLD
self = MPI.COMM_SELF
rank = comm.rank
size = comm.size

# Parameters
res = 2**8                         # Domain (grid) resolution
batch_size = 10                    # Batch Size
N_samples = 12*batch_size*20       # Number of input/output datapoints generated
verbose = True                     # Verbosity
petsc_options = {
    "ksp_type": "cg",              # Krylov Solver
    "pc_type": "hypre",            # Preconditioner
    "pc_hypre_type": "boomeramg",  # Hypre Type
    "ksp_rtol": 1e-10,             # Relative Tolerance
}


# %% 2. FEM Setup --------------------------------------------------------------------------------------------
# Defining Mesh and Function Space
domain = mesh.create_unit_square(self, res, res, mesh.CellType.quadrilateral)    # Rectangular elements on [0,1]^2
V = functionspace(domain, ('Q', 2))                          # Function space with Lagrange elements

t_dim = domain.topology.dim         # Topological Dimension
f_dim = domain.topology.dim - 1     # Facet Dimension

# Boundary Conditions
domain.topology.create_connectivity(f_dim, t_dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, f_dim, boundary_facets)

u_dirichlet = fem.Function(V)
u_dirichlet.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_dirichlet, boundary_dofs)

# Specifying Trial and Test Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# %% 3. Data Generation --------------------------------------------------------------------------------------
# Coordinates
x_coords = V.tabulate_dof_coordinates()[:, 0]
y_coords = V.tabulate_dof_coordinates()[:, 1]
num_nodes = x_coords.shape[0]

# Containers for Input/Output Data
inputs_local = []
outputs_local = []
batch_counter = 0

# Data Generation Loop
for i in range(rank, N_samples, size):
    if verbose:
        print(f"[Rank {rank}] Sample {i+1}/{N_samples}")

    # Generate Forcing Function
    f_expr = sample_field_2d(
        var=3,
        len_scale=1/3,
        smoothness=2,
        type='matern',
    )
    f = fem.Function(V)
    f.interpolate(lambda x: f_expr(x))

    # Generate Diffusion Coefficient
    a_expr = sample_field_2d(
        var=1,
        len_scale=1/3,
        smoothness=2,
        type='matern',
    )
    a = fem.Function(V)
    a.interpolate(lambda x: np.exp(a_expr(x)))

    # Variational Formulation
    A = ufl.inner(a * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Solve PDE
    problem = LinearProblem(
        A,                                 # Bilinear Form
        L,                                 # Linear Form (RHS)
        bcs=[bc],                          # Dirichlet Boundary Conditions
        petsc_options=petsc_options        # Solver Options
    )
    u_solve = problem.solve()

    # Gathering the functions
    u_vec, u_grid = gather_function_data(u_solve, V, comm)
    f_vec, f_grid = gather_function_data(f, V, comm)
    a_vec, a_grid = gather_function_data(a, V, comm)

    # Broadcast the gathered grids
    u_grid = comm.bcast(u_grid, root=0)
    f_grid = comm.bcast(f_grid, root=0)
    a_grid = comm.bcast(a_grid, root=0)

    if f_grid is not None and u_grid is not None and a_grid is not None:
        inp = np.stack([f_grid, a_grid], axis=0)      # shape (2, ny, nx)
        inputs_local.append(inp[None, ...])                  # shape (1, 2, ny, nx)
        outputs_local.append(u_grid[None, ...])              # shape (1, ny, nx)

    if len(inputs_local) >= batch_size:
        batch_inputs = np.concatenate(inputs_local, axis=0)
        batch_outputs = np.concatenate(outputs_local, axis=0)
        np.savez_compressed(f"darcy_batch_{rank}_rank_{batch_counter}.npz",
                            inputs=batch_inputs,
                            outputs=batch_outputs)

        # Clear and Increment
        inputs_local.clear()
        outputs_local.clear()
        batch_counter += 1

if verbose:
    print(
        f"✅ Rank {rank}: Data generation complete. "
        f"Saved {batch_counter} batch files "
        f"({batch_counter * batch_size} samples) "
        f"as darcy_batch_rank{rank}_*.npz."
    )