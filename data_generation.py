# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                                   Data Generation                                                    #
# -------------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------------------------
# 1. Poisson Equation
# ----------------------------------------------------------------------------------------------------------------------

# %% 1.1. Preamble -------------------------------------------------------------------------------------------
# Required Packages
import sys
import pandas as pd
import numpy as np
import ufl
import plotly.graph_objects as go

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

# %% 1.2. FEM Setup ------------------------------------------------------------------------------------------
# Defining mesh and function space
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    128,
    128,
    mesh.CellType.triangle
) # Triangular Elements on [0,1]^2
V = functionspace(domain, ("Lagrange", 1)) # Function space with Lagrange elements

# Boundary Conditions
t_dim = domain.topology.dim
f_dim = t_dim - 1
domain.topology.create_connectivity(f_dim, t_dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, f_dim, boundary_facets)

# Create a zero function in V
u_Dirichlet = fem.Function(V)
u_Dirichlet.interpolate(lambda x: np.zeros_like(x[0]))

bc = fem.dirichletbc(u_Dirichlet, boundary_dofs)

# Specifying Trial and Test Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# %% 1.3. Sampling Forcing -----------------------------------------------------------------------------------
# Forcing Function
def sample_f(L=10, M=10, alpha=2.0, scale=1.0):
    """
    Returns a forcing function on [0,1]^2 represented as a random Fourier series.

    Parameters:
      L, M  : int
              The number of Fourier modes to include in the x and y directions respectively.
      alpha : float
              The exponent controlling the decay of the coefficient variances.
              Larger alpha gives faster decay (smoother functions).
      scale : float
              An overall scaling factor for the forcing function.

    Returns:
      A function f(x) that accepts a dictionary or tuple-like structure of coordinate arrays x,
      where x[0] are the x-coordinates and x[1] the y-coordinates.
    """
    # Frequency indices for Dirichlet boundary conditions (sine expansions)
    n_vals = np.arange(1, L + 1)[:, None]
    m_vals = np.arange(1, M + 1)[:, None]

    # Calculating Variances that Decay
    sigma = 1.0 / (1.0 + (n_vals**2).reshape(L, 1) + (m_vals**2).reshape(1, M)) ** (alpha / 2)

    # Sample coefficients from a normal distribution
    coeffs = np.random.randn(L, M) * sigma

    def forcing_func(x):
        xx = x[0].reshape(1, -1)
        yy = x[1].reshape(1, -1)

        sine_x = np.sin(np.pi * n_vals * xx)
        sine_y = np.sin(np.pi * m_vals * yy)

        out = np.einsum('nm,ni,mi->i', coeffs, sine_x, sine_y)
        return scale * out

    return forcing_func

# Sample a random forcing function
f_sample = sample_f(L=10, M=10, alpha=2, scale=1)
f = fem.Function(V)
f.interpolate(lambda x: f_sample(x))

# Variational Formulation
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx


# %% 1.4. FEM Solving and Plotting ---------------------------------------------------------------------------
# Variational Formulation
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solving the Linear System
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_solve = problem.solve()

# Plotting
# Mesh Info
tdim = domain.topology.dim
topology, cell_types, geometry = vtk_mesh(domain, tdim)
geometry = geometry.reshape(-1, 3)
geometry_3d = geometry.copy()

# Extract cell connectivity for triangles
cells = topology.reshape(-1, 4)[:, 1:]

fig = go.Figure()

# Forcing
fig.add_trace(go.Mesh3d(
    x=geometry_3d[:, 0],
    y=geometry_3d[:, 1],
    z=f.x.array,
    i=cells[:, 0],
    j=cells[:, 1],
    k=cells[:, 2],
    intensity=f.x.array,
    colorscale="Viridis",
    showscale=True,
    flatshading=True,
    name="Forcing Function",
    opacity=0.75
))

fig.update_layout(
    title="Forcing Function",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="f(x, y)"
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()



fig = go.Figure()

# Forcing
fig.add_trace(go.Mesh3d(
    x=geometry_3d[:, 0],
    y=geometry_3d[:, 1],
    z=u_solve.x.array,
    i=cells[:, 0],
    j=cells[:, 1],
    k=cells[:, 2],
    intensity=u_solve.x.array,
    colorscale="Viridis",
    showscale=True,
    flatshading=True,
    name="FEM Solution",
    opacity=0.75
))


fig.update_layout(
    title="FEM Solution",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="u(x, y)"
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()

# %% 1.5. FEM Generation -------------------------------------------------------------------------------------
# Number of samples
N = 1000
inputs = []
outputs = []

x_coords = V.tabulate_dof_coordinates()[:, 0]
y_coords = V.tabulate_dof_coordinates()[:, 1]
num_nodes = x_coords.shape[0]
res = int(np.sqrt(num_nodes))

# Batch data generation
for i in range(N):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Sample {i+1}/{N}")

    # Generate a Forcing Function
    f_expr = sample_f()
    f = fem.Function(V)
    f.interpolate(lambda x: f_expr(x))

    # Assemble RHS
    L = f * v * ufl.dx

    # Solve PDE
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_solve = problem.solve()

    # Reshape and store
    inputs.append(f.x.array.reshape(res, res))
    outputs.append(u_solve.x.array.reshape(res, res))

# Save to .npy for FNO training
if MPI.COMM_WORLD.rank == 0:
    inputs = np.array(inputs)[:, None, :, :]
    outputs = np.array(outputs)[:, None, :, :]
    np.save("poisson_inputs.npy", inputs)
    np.save("poisson_outputs.npy", outputs)




# List to accumulate DataFrame for each sample
data_list = []

# Loop over N samples
for sample in range(N):
    print(f"Processing sample {sample + 1}/{N}...")
    # Generate a new random forcing function f for this sample.
    f_sample = sample_f(L=10, M=10, alpha=2, scale=1)
    f.interpolate(lambda x: f_sample(x))

    # Solve the FEM system with the new forcing.
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_solve = problem.solve()

    # Extract the forcing and solution values at the nodes.
    f_vals = f.x.array  # Forcing values (array of length equal to number of nodes)
    u_vals = u_solve.x.array  # FEM solution at each node

    # Build a DataFrame for this sample. We include a sample_id column to keep track.
    n_nodes = len(x_coords)
    df_sample = pd.DataFrame({
        "sample_id": [sample] * n_nodes,
        "node_id": range(n_nodes),
        "x": x_coords,
        "y": y_coords,
        "forcing": f_vals,
        "solution": u_vals
    })
    data_list.append(df_sample)

# Combine all the samples into one DataFrame and write it out to a CSV file.
df_all = pd.concat(data_list, ignore_index=True)
df_all.to_csv("PoissonEqn.csv", index=False)

