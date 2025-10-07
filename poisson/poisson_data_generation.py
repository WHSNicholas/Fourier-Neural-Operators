# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                        2D Poisson Equation - Data Generation                                         #
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
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from utils.data import sample_field_2d, gather_function_data
from petsc4py import PETSc

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
    "ksp_rtol": 1e-8,              # Relative Tolerance
}


# %% 2. FEM Setup --------------------------------------------------------------------------------------------
# Defining Mesh and Function Space
domain = mesh.create_unit_square(self, res, res, mesh.CellType.quadrilateral)      # Rectangular elements on [0,1]^2
V = functionspace(domain, ('Q', 2))                                          # Function space with Lagrange elements

t_dim = domain.topology.dim         # Topological Dimension
f_dim = domain.topology.dim - 1     # Facet Dimension

# Boundary Conditions
domain.topology.create_connectivity(f_dim, t_dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, f_dim, boundary_facets)

u_dirichlet = fem.Function(V)
u_dirichlet.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

# Specifying Trial and Test Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Stiffness Matrix
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
A = assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()

# Solution
u_solve = fem.Function(V)

# Solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setFromOptions()

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

    f_expr = sample_field_2d(
        res=res,
        var=3,
        len_scale=1/3,
        smoothness=2,
        type='matern',
    )
    f = fem.Function(V)
    f.interpolate(lambda x: f_expr(x))

    # Assemble RHS and solve PDE
    L = f * v * ufl.dx
    L = assemble_vector(fem.form(L))
    fem.apply_lifting(L, [fem.form(a)], bcs=[[bc]])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, [bc])

    # Solve
    solver.solve(L, u_solve.x.petsc_vec)
    u_solve.x.scatter_forward()

    # Gathering the functions
    u_vec, u_grid = gather_function_data(u_solve, V, comm)
    f_vec, f_grid = gather_function_data(f, V, comm)

    # Broadcast the gathered grids
    u_grid = comm.bcast(u_grid, root=0)
    f_grid = comm.bcast(f_grid, root=0)

    if f_grid is not None and u_grid is not None:
        inputs_local.append(f_grid[None, :, :])
        outputs_local.append(u_grid[None, :, :])

    if len(inputs_local) >= batch_size:
        batch_inputs = np.concatenate(inputs_local, axis=0)  # Shape (batch_size, H, W)
        batch_outputs = np.concatenate(outputs_local, axis=0)
        np.savez_compressed(f"poisson_batch_rank{rank}_{batch_counter}.npz",
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
        f"as poisson_batch_rank{rank}_*.npz."
    )