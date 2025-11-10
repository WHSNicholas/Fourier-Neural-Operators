# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                  1D Black-Scholes Equation - Data Generation                                        #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dependencies
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.data import sample_field_1d
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Parameters
d_res = 2 ** 5  # Domain Resolution
S_min = 1e-6  # Minimum stock price
S_max = 300.0  # Maximum stock price
T = 1.0  # Final Time (1 year)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

batch_size = 120  # Batch Size
N_samples = batch_size * 20  # Total samples (2400)
verbose = True  # Verbosity


# %% 2. Finite Difference Setup -----------------------------------------------------------------------------
def setup_black_scholes_grid(S_min, S_max, d_res, T, N_t=500):
    """Set up non-uniform grid for Black-Scholes"""
    # Create non-uniform grid with clustering near typical strike regions
    S_grid = np.zeros(d_res)

    # Use hyperbolic tangent stretching for better resolution around S=100
    center = 100.0
    scale = 50.0
    uniform_grid = np.linspace(-1, 1, d_res)

    for i in range(d_res):
        S_grid[i] = center + scale * np.tanh(1.5 * uniform_grid[i])

    # Ensure boundaries
    S_grid[0] = S_min
    S_grid[-1] = S_max
    S_grid = np.sort(S_grid)

    # Time grid (finer near expiry)
    t_grid = np.linspace(0, T, N_t)

    return S_grid, t_grid


def create_black_scholes_operator(S_grid, r, sigma, dt, theta):
    """Create finite difference operator for Black-Scholes"""
    N = len(S_grid)
    alpha = np.zeros(N)
    beta = np.zeros(N)
    gamma = np.zeros(N)

    # Central differences for interior points
    for i in range(1, N - 1):
        dS_plus = S_grid[i + 1] - S_grid[i]
        dS_minus = S_grid[i] - S_grid[i - 1]
        dS_center = 0.5 * (dS_plus + dS_minus)

        # First derivative coefficients
        dVdS = (S_grid[i + 1] - S_grid[i - 1]) / (dS_plus + dS_minus)

        # Second derivative coefficients
        d2VdS2_plus = 2.0 / (dS_plus * (dS_plus + dS_minus))
        d2VdS2_minus = 2.0 / (dS_minus * (dS_plus + dS_minus))
        d2VdS2_center = -2.0 / (dS_plus * dS_minus)

        # Black-Scholes coefficients
        alpha[i] = 0.5 * sigma ** 2 * S_grid[i] ** 2 * d2VdS2_minus - r * S_grid[i] / (2 * dS_center)
        beta[i] = 0.5 * sigma ** 2 * S_grid[i] ** 2 * d2VdS2_center - r
        gamma[i] = 0.5 * sigma ** 2 * S_grid[i] ** 2 * d2VdS2_plus + r * S_grid[i] / (2 * dS_center)

    # Build matrices
    main_diag = 1.0 - theta * dt * beta
    lower_diag = -theta * dt * alpha[1:]
    upper_diag = -theta * dt * gamma[:-1]

    # Boundary conditions
    main_diag[0] = 1.0  # Dirichlet at S_min
    main_diag[-1] = 1.0  # Neumann at S_max
    lower_diag[0] = 0.0
    upper_diag[-1] = 0.0

    A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')

    # Right-hand side matrix
    main_diag_rhs = 1.0 + (1 - theta) * dt * beta
    lower_diag_rhs = (1 - theta) * dt * alpha[1:]
    upper_diag_rhs = (1 - theta) * dt * gamma[:-1]

    main_diag_rhs[0] = 1.0
    main_diag_rhs[-1] = 1.0
    lower_diag_rhs[0] = 0.0
    upper_diag_rhs[-1] = 0.0

    M = sparse.diags([lower_diag_rhs, main_diag_rhs, upper_diag_rhs], [-1, 0, 1], format='csc')

    return A, M


def solve_black_scholes(payoff, S_grid, T, r, sigma, N_t=500):
    """Solve Black-Scholes PDE using θ-scheme with Rannacher smoothing"""
    dt = T / N_t
    V = payoff.copy()

    # Rannacher smoothing: first 4 steps with Implicit Euler (θ=1.0)
    for step in range(min(4, N_t)):
        A, M = create_black_scholes_operator(S_grid, r, sigma, dt, theta=1.0)
        rhs = M.dot(V)

        # Boundary conditions
        rhs[0] = payoff[0] * np.exp(-r * (step + 1) * dt)  # Dirichlet at S_min
        rhs[-1] = V[-1]  # Neumann at S_max

        V = spla.spsolve(A, rhs)

    # Remaining steps with θ=0.55
    for step in range(4, N_t):
        A, M = create_black_scholes_operator(S_grid, r, sigma, dt, theta=0.55)
        rhs = M.dot(V)

        # Boundary conditions
        rhs[0] = payoff[0] * np.exp(-r * (step + 1) * dt)  # Dirichlet at S_min
        rhs[-1] = V[-1]  # Neumann at S_max

        V = spla.spsolve(A, rhs)

    return V


# %% 3. Data Generation --------------------------------------------------------------------------------------
print("Setting up Black-Scholes grid...")
S_grid, t_grid = setup_black_scholes_grid(S_min, S_max, d_res, T)

inputs = []
outputs = []

for i in range(N_samples):
    if verbose and (i % 100 == 0):
        print(f"Generating sample {i + 1}/{N_samples}")

    # Sample terminal payoff using 1D Gaussian field (Matern field)
    payoff_expr = sample_field_1d(res=d_res, L=1.0, type='matern',
                                  smoothness=1 / 3, nu=2, var=3)

    # Generate payoff on normalized coordinates and map to S-grid
    x_normalized = (S_grid - S_min) / (S_max - S_min)
    payoff = torch.from_numpy(payoff_expr((x_normalized,))).float().numpy()

    # Ensure payoff is non-negative (financial constraint)
    payoff = np.maximum(payoff, 0)

    # Solve Black-Scholes PDE to get price at time 0
    try:
        price_at_t0 = solve_black_scholes(payoff, S_grid, T, r, sigma)

        inputs.append(payoff)
        outputs.append(price_at_t0)

    except Exception as e:
        if verbose:
            print(f"Error in sample {i}: {e}")
        continue

    # Save batches
    if len(inputs) >= batch_size:
        batch_id = len(inputs) // batch_size
        batch_inputs = np.stack(inputs[:batch_size])
        batch_outputs = np.stack(outputs[:batch_size])

        np.savez_compressed(f"black_scholes_batch_{batch_id}.npz",
                            inputs=batch_inputs,
                            outputs=batch_outputs,
                            S_grid=S_grid)

        # Remove saved batch
        inputs = inputs[batch_size:]
        outputs = outputs[batch_size:]

# Save any remaining samples
if len(inputs) > 0:
    batch_id = (N_samples // batch_size) + 1
    np.savez_compressed(f"black_scholes_batch_{batch_id}.npz",
                        inputs=np.stack(inputs),
                        outputs=np.stack(outputs),
                        S_grid=S_grid)

if verbose:
    print(
        f"✅ Black-Scholes data generation complete. "
        f"Saved {batch_id} batch files "
        f"({N_samples} samples) "
        f"as black_scholes_batch_*.npz"
    )

    # Print sample statistics
    sample_input = inputs[0] if len(inputs) > 0 else np.load(f"black_scholes_batch_1.npz")['inputs'][0]
    sample_output = outputs[0] if len(outputs) > 0 else np.load(f"black_scholes_batch_1.npz")['outputs'][0]

    print(f"Sample payoff range: [{np.min(sample_input):.3f}, {np.max(sample_input):.3f}]")
    print(f"Sample price range: [{np.min(sample_output):.3f}, {np.max(sample_output):.3f}]")
    print(f"Grid size: {d_res}, S range: [{S_min:.1f}, {S_max:.1f}]")