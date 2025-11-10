# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                    1D Inviscid Burgers Equation - Data Generation                                    #
# -------------------------------------------------------------------------------------------------------------------- #


# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.data import sample_field_1d, legendre, dlegendre, limiter, compute_dt, rhs, reconstruct_solution
from numpy.polynomial.legendre import leggauss
import time
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# Parameters
d_res = 2 ** 11                    # Domain Resolution
L = 1                              # Domain Length
T = 0.3                            # Final Time
dx = L / d_res                     # Grid Increments
poly_order = 1                     # Polynomial elements
quad_order = 2 * poly_order + 1    # Quadrature Order
cfl = 0.25                         # CFL

# Data Generation Parameters
batch_size = 120                   # Batch Size
N_samples = batch_size * 20        # Number of input/output datapoints generated
verbose = True                     # Verbosity
save_frequency = 10                # Save intermediate states every N timesteps
output_resolution = d_res          # Resolution for output data

# Random seed
np.random.seed(2025)


# %% 2. Discontinuous Galerkin Setup -------------------------------------------------------------------------
# Flux Function
def flux(u):
    return 0.5 * u ** 2

# Create Mesh
x_edges = np.linspace(0, L, d_res + 1)
x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])

# Gauss-Legendre Quadrature
quad_points, quad_weights = leggauss(quad_order)

# Precompute all basis matrices (these are constant for all simulations)
PHI = np.zeros((quad_order, poly_order + 1))
DPHI = np.zeros((quad_order, poly_order + 1))
for k in range(poly_order + 1):
    PHI[:, k] = legendre(quad_points, k)
    DPHI[:, k] = dlegendre(quad_points, k)

WDPHI = quad_weights[:, None] * DPHI
PHI_L = np.array([legendre(-1.0, k) for k in range(poly_order + 1)])
PHI_R = np.array([legendre(1.0, k) for k in range(poly_order + 1)])
M_inv = (2 * np.arange(poly_order + 1) + 1) / dx

PROJ = np.zeros((poly_order + 1, quad_order))
for k in range(poly_order + 1):
    PROJ[k, :] = (2 * k + 1) / 2 * quad_weights * PHI[:, k]

# Output grid for saving solutions
x_output = np.linspace(0, L, output_resolution)


# %% 3. Solver Function --------------------------------------------------------------------------------------
def solve_burgers_dg(u0_func, T_final, save_times=None, verbose_solver=False):
    """
    Solve the 1D inviscid Burgers equation using DG method.

    Args:
        u0_func: Initial condition function (callable)
        T_final: Final simulation time
        save_times: List of times at which to save the solution
        verbose_solver: Print solver progress

    Returns:
        solutions: List of solutions at save_times
        actual_times: Actual times corresponding to saved solutions
    """
    # Initialize solution
    x_local_all = x_edges[:-1, None] + 0.5 * (quad_points[None, :] + 1) * dx
    u_local_all = np.asarray(u0_func((x_local_all.ravel(),))).reshape(x_local_all.shape)
    u_coeffs = u_local_all @ PROJ.T

    # Time integration
    t = 0
    step = 0

    # Storage for solutions at requested times
    if save_times is None:
        save_times = [T_final]

    solutions = []
    actual_times = []
    save_idx = 0

    while t < T_final and save_idx < len(save_times):
        # Adaptive timestep
        dt = compute_dt(u_coeffs, dx, cfl, PHI, PHI_L, PHI_R)

        # Check if we need to save at this step
        if save_idx < len(save_times) and t + dt >= save_times[save_idx]:
            # Adjust dt to hit the save time exactly
            dt = save_times[save_idx] - t

            # Time integration step
            # SSP-RK3 Stage 1
            rhs1 = rhs(u_coeffs, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u1 = u_coeffs + dt * rhs1
            u1 = limiter(u1, dx)

            # SSP-RK3 Stage 2
            rhs2 = rhs(u1, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u2 = 0.75 * u_coeffs + 0.25 * (u1 + dt * rhs2)
            u2 = limiter(u2, dx)

            # SSP-RK3 Stage 3
            rhs3 = rhs(u2, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u_coeffs = (1 / 3) * u_coeffs + (2 / 3) * (u2 + dt * rhs3)
            u_coeffs = limiter(u_coeffs, dx)

            # Save solution
            u_solution = reconstruct_solution(u_coeffs, x_output, x_edges, dx, legendre, d_res)
            solutions.append(u_solution)
            actual_times.append(save_times[save_idx])
            save_idx += 1

        else:
            # Normal time integration step
            dt = min(dt, T_final - t)

            # SSP-RK3 Stage 1
            rhs1 = rhs(u_coeffs, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u1 = u_coeffs + dt * rhs1
            u1 = limiter(u1, dx)

            # SSP-RK3 Stage 2
            rhs2 = rhs(u1, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u2 = 0.75 * u_coeffs + 0.25 * (u1 + dt * rhs2)
            u2 = limiter(u2, dx)

            # SSP-RK3 Stage 3
            rhs3 = rhs(u2, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv)
            u_coeffs = (1 / 3) * u_coeffs + (2 / 3) * (u2 + dt * rhs3)
            u_coeffs = limiter(u_coeffs, dx)

        t += dt
        step += 1

        # Check for NaNs
        if np.any(np.isnan(u_coeffs)):
            if verbose_solver:
                print(f"WARNING: NaN detected at time {t}, step {step}")
            break

        if verbose_solver and step % 100 == 0:
            print(f"  Step {step}: t = {t:.4f}, dt = {dt:.2e}")

    # If we haven't saved the final solution yet
    while save_idx < len(save_times):
        u_solution = reconstruct_solution(u_coeffs, x_output, x_edges, dx, legendre, d_res)
        solutions.append(u_solution)
        actual_times.append(t)
        save_idx += 1

    return solutions, actual_times


# %% 4. Data Generation Loop ---------------------------------------------------------------------------------
def generate_data_batch(n_samples, batch_id):
    """
    Generate a batch of training data.

    Args:
        n_samples: Number of samples to generate
        batch_id: Batch identifier

    Returns:
        inputs: Array of initial conditions
        outputs: Array of solutions at time T
    """

    inputs = []
    outputs = []
    metadata = []


    print(f"\nGenerating batch {batch_id} with {n_samples} samples...")

    for i in tqdm(range(n_samples), desc=f"Batch {batch_id}"):
        # Generate random initial condition
        # # You can vary these parameters to create diverse initial conditions
        # smoothness = np.random.uniform(0.1, 0.5)
        # amplitude = np.random.uniform(0.5, 2.0)
        # nu = np.random.uniform(1.0, 2.0)
        # var = np.random.uniform(0.5, 2.0)

        # Sample initial condition
        u0_func = sample_field_1d(
            res=output_resolution,
            L=L,
            type='matern',
            smoothness=1/3,
            amplitude=1,
            nu=2,
            var=3
        )

        # Get initial condition on output grid
        u_initial = np.asarray(u0_func((x_output,))).reshape(x_output.shape)

        try:
            # Solve the PDE
            solutions, times = solve_burgers_dg(
                u0_func=u0_func,
                T_final=T,
                save_times=[T],  # Only save final time for training
                verbose_solver=False
            )

            # Store input/output pair
            inputs.append(u_initial)
            outputs.append(solutions[0])

            # # Store metadata for reproducibility
            # metadata.append({
            #     'smoothness': smoothness,
            #     'amplitude': amplitude,
            #     'nu': nu,
            #     'var': var,
            #     'final_time': times[0]
            # })

        except Exception as e:
            print(f"\nError in sample {i}: {e}")
            # Generate a simple fallback initial condition
            u_initial = np.sin(2 * np.pi * x_output)
            inputs.append(u_initial)
            outputs.append(u_initial)  # Fallback: no evolution
            metadata.append({'error': str(e)})

    # Convert to arrays
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)

    # Save the batch
    filename = f"inviscid_burgers_batch_{batch_id}.npz"
    np.savez_compressed(
        filename,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        x_grid=x_output,
        T=T,
        d_res=d_res,
        poly_order=poly_order
    )

    print(f"✅ Batch {batch_id} saved to {filename}")

    return inputs, outputs


if __name__ == "__main__":
    print("=" * 80)
    print("DG BURGERS EQUATION - DATA GENERATION")
    print("=" * 80)
    print(f"Domain resolution (DG): {d_res}")
    print(f"Output resolution: {output_resolution}")
    print(f"Polynomial order: {poly_order}")
    print(f"Final time: {T}")
    print(f"Total samples: {N_samples}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    # Timer
    start_time = time.time()

    # Generate data in batches
    all_inputs = []
    all_outputs = []

    num_batches = N_samples // batch_size

    for batch_id in range(1, num_batches + 1):
        inputs, outputs = generate_data_batch(
            n_samples=batch_size,
            batch_id=batch_id
        )

        all_inputs.append(inputs)
        all_outputs.append(outputs)

        # Optional: visualize a sample from each batch
        if verbose:
            plt.figure(figsize=(12, 4))

            # Plot first sample from batch
            plt.subplot(1, 3, 1)
            plt.plot(x_output, inputs[0], 'b-', label='Initial')
            plt.plot(x_output, outputs[0], 'r-', label=f'T={T}')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.title(f'Batch {batch_id}, Sample 1')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot middle sample from batch
            mid_idx = batch_size // 2
            plt.subplot(1, 3, 2)
            plt.plot(x_output, inputs[mid_idx], 'b-', label='Initial')
            plt.plot(x_output, outputs[mid_idx], 'r-', label=f'T={T}')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.title(f'Batch {batch_id}, Sample {mid_idx + 1}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot statistics
            plt.subplot(1, 3, 3)
            plt.hist(outputs.flatten(), bins=50, alpha=0.7, density=True)
            plt.xlabel('u values')
            plt.ylabel('Density')
            plt.title(f'Output Distribution (Batch {batch_id})')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'batch_{batch_id}_samples.png', dpi=100)

    # Combine all batches
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Save combined dataset
    np.savez_compressed(
        'inviscid_burgers_complete.npz',
        inputs=all_inputs,
        outputs=all_outputs,
        x_grid=x_output,
        T=T,
        d_res=d_res,
        poly_order=poly_order
    )

    # Print summary statistics
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"✅ Generated {N_samples} samples in {num_batches} batches")
    print(f"✅ Total time: {elapsed_time:.2f} seconds")
    print(f"✅ Average time per sample: {elapsed_time / N_samples:.3f} seconds")
    print("\nDataset Statistics:")
    print(f"  Input shape: {all_inputs.shape}")
    print(f"  Output shape: {all_outputs.shape}")
    print(f"  Input range: [{all_inputs.min():.3f}, {all_inputs.max():.3f}]")
    print(f"  Output range: [{all_outputs.min():.3f}, {all_outputs.max():.3f}]")
    print("=" * 80)


