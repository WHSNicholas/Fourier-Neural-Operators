# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                                      Darcy Flow                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------------------------
# 1. Darcy Flow Example
# ----------------------------------------------------------------------------------------------------------------------

# %% 1.1. Preamble -------------------------------------------------------------------------------------------
# Required Packages
import torch
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import ufl
import plotly.graph_objects as go

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

# Constants
DEVICE = 'cpu'

# %% 1.2. Loading Data ---------------------------------------------------------------------------------------
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(DEVICE)


# %% 1.3. Simple FNO Model -----------------------------------------------------------------------------------
model = FNO(n_modes=(16, 16),
             in_channels=1,
             out_channels=1,
             hidden_channels=32,
             projection_channel_ratio=2)
model = model.to(DEVICE)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %% 1.4. Training Model -------------------------------------------------------------------------------------
# Optimiser
optimizer = AdamW(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

trainer = Trainer(model=model, n_epochs=20,
                  device=DEVICE,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

# %% 1.5. Model Visualisation --------------------------------------------------------------------------------
test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction (16x16).', y=0.98)
plt.tight_layout()
fig.show()

# Zero Shot Super Evaluation
test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction (32x32).', y=0.98)
plt.tight_layout()
fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# 2. Simple Forward Implementation
# ----------------------------------------------------------------------------------------------------------------------
# %% 2.1. Creating FNO ---------------------------------------------------------------------------------------
class FNO1D:
    def __init__(self, d_a=2, d_v=8, d_u=1, n_points=64):
        """
        A single-layer 1D Fourier Neural Operator.

        Args:
            d_a: Dimension (number of channels) of the input a(x).
            d_v: Number of channels in the hidden (lifted) representation.
            d_u: Number of channels in the final output u(x).
            n_points: Number of spatial points for discretizing the domain [0,1].
        """
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.n_points = n_points

        # Lift: R is (d_v x d_a)
        self.R = np.random.randn(d_v, d_a)

        # Local linear transform W: (d_v x d_v) plus bias b: (d_v,)
        self.W = np.random.randn(d_v, d_v)
        self.b = np.random.randn(d_v)

        # Fourier kernel: P[k] is a (d_v x d_v) matrix for each frequency k
        # For simplicity, store all frequency modes from 0..(n_points-1).
        # A real FNO may store fewer modes or handle negative frequencies carefully.
        self.P = np.random.randn(n_points, d_v, d_v) + 1j * np.random.randn(n_points, d_v, d_v)

        # Projection Q: (d_u x d_v)
        self.Q = np.random.randn(d_u, d_v)

    def relu(self, x):
        return np.maximum(x, 0.0)

    def forward(self, a):
        """
        Forward pass for a single sample a(x).

        Args:
            a: np.ndarray of shape (n_points, d_a), the input function discretized at n_points.

        Returns:
            u: np.ndarray of shape (n_points, d_u), the output function.
        """
        # 1. Lifting Operator
        # Shape of a: (n_points, d_a)
        # After lifting: (n_points, d_v)
        v = a @ self.R.T  # matrix multiply each point in space

        # 2. Fourier Layer
        # a. Compute FFT along the spatial dimension for each channel
        #    We'll do an FFT for each of the d_v channels
        #    v_freq shape: (n_points, d_v) in complex domain
        v_freq = np.fft.fft(v, axis=0)

        # b. Multiply by trainable Fourier kernel in frequency space
        #    For each frequency k, we have a (d_v x d_v) matrix P[k].
        #    So we do: v_freq[k] = P[k] * v_freq[k].
        #    v_freq[k] and v_freq[k, :] are each shape (d_v).
        #    P[k] is shape (d_v x d_v).
        v_freq_new = np.zeros_like(v_freq, dtype=np.complex128)
        for k in range(self.n_points):
            # (d_v,) = (d_v, d_v) dot (d_v,)
            v_freq_new[k] = self.P[k] @ v_freq[k]

        # c. Inverse FFT back to real space
        v_ifft = np.fft.ifft(v_freq_new, axis=0).real  # shape: (n_points, d_v)

        # d. Add local linear transform W, bias b, apply ReLU
        # shape remains (n_points, d_v)
        v_out = (v_ifft @ self.W.T) + self.b  # (n_points, d_v)
        v_out = self.relu(v_out)

        # 3. Projection
        # map from d_v channels down to d_u
        # shape: (n_points, d_u)
        u = v_out @ self.Q.T

        return v, v_freq, v_freq_new, v_ifft, v_out, u

# %% 2.2. Example Usage --------------------------------------------------------------------------------------
n_points = 64
d_a = 2
d_v = 8
d_u = 1

# Create random "input function" a(x): shape (64, 2)
# In practice, you'd load real PDE data here
a_sample = np.random.randn(n_points, d_a)

# Instantiate the FNO1D
fno_layer = FNO1D(d_a=d_a, d_v=d_v, d_u=d_u, n_points=n_points)

# Forward pass
v, v_freq, v_freq_new, v_ifft, v_out, u_pred = fno_layer.forward(a_sample)

print("Input shape:", a_sample.shape)  # (64, 2)
print("Output shape:", u_pred.shape)  # (64, 1)
print("Sample output (first 5 points):\n", u_pred[:5])

plt.plot(a_sample)
plt.show()

# %% 2.3. Understanding Fourier Transforms -------------------------------------------------------------------
# Create a function in the spatial domain
n_points = 128  # Number of spatial points
x = np.linspace(0, 1, n_points)  # Spatial domain (normalized 0 to 1)
a = np.sin(2 * np.pi * 4 * x) + 0.5 * np.sin(2 * np.pi * 8 * x)  # Sum of sinusoids

# Fourier Transform
a_freq = np.fft.fft(a)  # FFT of the function
freqs = np.fft.fftfreq(n_points, d=(x[1] - x[0]))  # Frequency axis

# Apply a simple Fourier operator (trainable kernel)
# Example: Damp high frequencies
P_k = np.exp(-0.1 * np.abs(freqs))  # Simple decaying function for kernel
a_freq_transformed = P_k * a_freq  # Apply kernel

# Inverse Fourier Transform
a_transformed = np.fft.ifft(a_freq_transformed).real  # Back to real space

# === PLOTS ===
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# 1. Plot original function in real space
ax[0].plot(x, a, label="Original Function a(x)")
ax[0].set_title("Original Function in Spatial Domain")
ax[0].legend()

# 2. Plot magnitude of Fourier coefficients
ax[1].stem(freqs[:n_points//2], np.abs(a_freq[:n_points//2]))
ax[1].set_title("Fourier Transform (Magnitude Spectrum)")
ax[1].set_xlabel("Frequency")

# 3. Plot transformed function after modifying Fourier coefficients
ax[2].plot(x, a_transformed, label="Modified Function (Inverse FFT)")
ax[2].set_title("Function After Fourier Transformation in FNO")
ax[2].legend()

plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------------------------------------------------
# 3. Galerkin FEM for Poisson's Equation - First Principles
# ----------------------------------------------------------------------------------------------------------------------
# Mesh resolution
N = 100
h = 1.0 / N
nodes = []
elements = []

# Generate nodes
for i in range(N + 1):
    for j in range(N + 1):
        nodes.append((i * h, j * h))
nodes = np.array(nodes)

# Generate elements: split each square into two triangles
for i in range(N):
    for j in range(N):
        n0 = i * (N + 1) + j
        n1 = n0 + 1
        n2 = n0 + (N + 1)
        n3 = n2 + 1
        elements.append((n0, n1, n3))  # lower triangle
        elements.append((n0, n3, n2))  # upper triangle

# Define source function f(x, y)
def f(x, y):
    return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Initialize stiffness matrix and load vector
num_nodes = (N + 1) ** 2
A = lil_matrix((num_nodes, num_nodes))
b = np.zeros(num_nodes)

# Assemble global matrix and vector
for element in elements:
    indices = np.array(element)
    verts = nodes[indices]

    x = verts[:, 0]
    y = verts[:, 1]

    # Triangle area
    area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    # Gradients of basis functions
    mat = np.ones((3, 3))
    mat[:, 1] = x
    mat[:, 2] = y
    grads = np.linalg.inv(mat)[:, 1:]  # gradients of basis functions

    # Stiffness matrix contributions
    for i in range(3):
        for j in range(3):
            A[indices[i], indices[j]] += area * np.dot(grads[i], grads[j])

    # Load vector (midpoint quadrature)
    x_m = np.mean(x)
    y_m = np.mean(y)
    f_val = f(x_m, y_m)
    for i in range(3):
        b[indices[i]] += f_val * area / 3

# Apply Dirichlet boundary conditions
boundary_nodes = set()
for i in range(N + 1):
    boundary_nodes.add(i)  # bottom
    boundary_nodes.add(i + N * (N + 1))  # top
    boundary_nodes.add(i * (N + 1))  # left
    boundary_nodes.add(i * (N + 1) + N)  # right

interior_nodes = list(set(range(num_nodes)) - boundary_nodes)

# Reduce system
A_int = A[interior_nodes][:, interior_nodes].tocsc()
b_int = b[interior_nodes]

# Solve the linear system
u_int = spsolve(A_int, b_int)

# Assemble full solution vector
u = np.zeros(num_nodes)
u[interior_nodes] = u_int

# Reshape for plotting
X = nodes[:, 0].reshape(N + 1, N + 1)
Y = nodes[:, 1].reshape(N + 1, N + 1)
U = u.reshape(N + 1, N + 1)

# Plot the FEM solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_title('FEM Solution to Poisson Equation')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 4. Galerkin FEM for Poisson's Equation - FEniCS
# ----------------------------------------------------------------------------------------------------------------------
# %% 4.1. PDE/FEM Setup --------------------------------------------------------------------------------------
# Defining mesh and function space
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    128,
    128,
    mesh.CellType.triangle
) # Triangular Elements on [0,1]^2
V = functionspace(domain, ("Lagrange", 1)) # Function space with Lagrange elements

# Defining Analytic Solution
u_Dirichlet = fem.Function(V)
u_Dirichlet.interpolate(lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Create facet to cell connectivity required to determine boundary facets
t_dim = domain.topology.dim
f_dim = t_dim - 1
domain.topology.create_connectivity(f_dim, t_dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Boundary Conditions
boundary_dofs = fem.locate_dofs_topological(V, f_dim, boundary_facets)
bc = fem.dirichletbc(u_Dirichlet, boundary_dofs)

# %% 4.2. Variational Formulation ----------------------------------------------------------------------------
# Specifying Test Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Forcing Function
f = fem.Function(V)
f.interpolate(lambda x: 4 * np.pi**4 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Variational Formulation
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solving the Linear System
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_solve = problem.solve()

# %% 4.3. Errors and Plots -----------------------------------------------------------------------------------
# Defining the Exact Solution
V2 = fem.functionspace(domain, ("Lagrange", 2))
u_exact = fem.Function(V2)
u_exact.interpolate(lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Computing the Error
L2_error = fem.form(ufl.inner(u_solve - u_exact, u_solve - u_exact) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = np.max(np.abs(u_Dirichlet.x.array-u_solve.x.array))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Plotting
# Exact Solution
u_exact_proj = fem.Function(V)
u_exact_proj.interpolate(lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Mesh Info
tdim = domain.topology.dim
topology, cell_types, geometry = vtk_mesh(domain, tdim)

# Ensure geometry is reshaped appropriately (typically 3D)
geometry = geometry.reshape(-1, 3)
geometry_3d = geometry.copy()

# Extract cell connectivity for triangles
cells = topology.reshape(-1, 4)[:, 1:]

fig = go.Figure()

# FEM numerical solution (P1)
fig.add_trace(go.Mesh3d(
    x=geometry_3d[:, 0],
    y=geometry_3d[:, 1],
    z=u_solve.x.array,      # FEM solution (P1)
    i=cells[:, 0],
    j=cells[:, 1],
    k=cells[:, 2],
    intensity=u_solve.x.array,
    colorscale="Viridis",
    showscale=True,
    flatshading=True,
    name="FEM Solution",
    opacity=0.3
))

# Analytical solution (projected onto P1)
fig.add_trace(go.Mesh3d(
    x=geometry_3d[:, 0],
    y=geometry_3d[:, 1],
    z=u_exact_proj.x.array,  # Now using projected values in the same function space as u_solve
    i=cells[:, 0],
    j=cells[:, 1],
    k=cells[:, 2],
    intensity=u_exact_proj.x.array,
    colorscale="Reds",
    showscale=False,
    flatshading=True,
    name="Analytical Solution (Projected)",
    opacity=0.3
))

fig.update_layout(
    title="FEM vs Analytical Solution on Unit Square",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="u(x, y)"
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()