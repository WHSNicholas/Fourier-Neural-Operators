# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                    1D Viscous Burgers Equation - Data Generation                                     #
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
from mpi4py import MPI

# Parallel Computing
comm = MPI.COMM_WORLD
self = MPI.COMM_SELF
rank = comm.rank
size = comm.size

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Parameters
d_res = 2**12                      # Domain Resolution
L = 1                              # Domain Length
dx = L / d_res                     # Grid Increments
T = 0.3                            # Final Time
nu = 0.1                           # Viscosity
CFL = 0.5                          # CFL

batch_size = 10                    # Batch Size
N_samples = 12*batch_size*20       # Number of input/output datapoints generated
verbose = True                     # Verbosity


# %% 2. Pseudospectral Setup ---------------------------------------------------------------------------------
# Defining Mesh and Function Space
x = torch.linspace(0, L, d_res, device=device)
k = 2 * np.pi * torch.fft.rfftfreq(d_res, d=dx).to(device)

# %% 3. Data Generation --------------------------------------------------------------------------------------
inputs_local = []
outputs_local = []
batch_counter = 0

for i in range(rank, N_samples, size):
    if verbose:
        print(f"[Rank {rank}] Sample {i+1}/{N_samples}")

    # Sample Initial Condition
    u0_expr = sample_field_1d(res=d_res, L=L, type='matern',
                              smoothness=1/3, nu=2, var=3)
    u0 = torch.from_numpy(u0_expr((x.cpu().numpy(), ))).to(device)
    u1 = u0.clone()

    t = 0
    while t < T - 1e-15:
        umax = float(torch.max(torch.abs(u1)).detach().cpu())
        # Calculating dt
        dt = CFL * dx / max(umax, 1e-8)
        if t + dt > T:
            dt = T - t
        t += dt

        # 1. Diffusion
        exp_diff = torch.exp(-nu * k ** 2 * dt)
        u_hat = torch.fft.rfft(u1)
        u1 = torch.fft.irfft(u_hat * exp_diff, n=d_res)

        # 2. Convection
        u_hat = torch.fft.rfft(u1)
        u_sq_hat = torch.fft.rfft(0.5 * u1 ** 2)
        N_cutoff = u_sq_hat.shape[-1] * 2 // 3
        u_sq_hat[N_cutoff:] = 0

        u_hat = u_hat - dt * (1j * k) * u_sq_hat
        u1 = torch.fft.irfft(u_hat, n=d_res)


    inputs_local.append(u0.cpu().numpy())
    outputs_local.append(u1.cpu().numpy())

    if len(inputs_local) >= batch_size:
        np.savez_compressed(os.path.join(f"burgers_batch_rank{rank}_{batch_counter}.npz"),
                            inputs=np.stack(inputs_local),
                            outputs=np.stack(outputs_local))
        inputs_local.clear()
        outputs_local.clear()
        batch_counter += 1

if len(inputs_local) > 0:
    np.savez_compressed(
        os.path.join("burgers_batches", f"burgers_batch_rank{rank}_{batch_counter}.npz"),
        inputs=np.stack(inputs_local),
        outputs=np.stack(outputs_local),
    )
    inputs_local.clear()
    outputs_local.clear()
    batch_counter += 1

if verbose:
    print(
        f"✅ Rank {rank}: Data generation complete. "
        f"Saved {batch_counter} batch files "
        f"({batch_counter * batch_size} samples) "
        f"as burgers/burgers_batch_rank{rank}_*.npz"
    )

