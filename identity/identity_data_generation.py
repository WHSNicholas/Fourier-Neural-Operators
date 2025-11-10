# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                       1D Identity Operator - Data Generation                                         #
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
d_res = 2**13                      # Domain resolution
L = 1.0                            # Domain length
dx = L / d_res                     # Grid spacing
low_cutoff = 2**5                  # Highest mode the FNO can represent (bandlimit)
high_min = low_cutoff + 1          # Start of high-frequency range
high_max = 2 * low_cutoff          # End of high-frequency range

# Data parameters
batch_size = 10                    # Batch size per save
N_samples = 12 * batch_size * 20   # Total number of samples
verbose = True                     # Verbosity flag
save_dir = "identity/data"      # Save directory
os.makedirs(save_dir, exist_ok=True)


# %% 2. Mesh and utility functions --------------------------------------------------------------------------
x = torch.linspace(0, L, d_res, device=device)

def sample_highfreq_signal():
    """
    Sample a 1D signal composed of random high-frequency sine waves.
    """
    n_modes = np.random.randint(1, 5)  # Random number of sine components
    u = torch.zeros_like(x)
    for _ in range(n_modes):
        freq = np.random.randint(high_min, high_max)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.0)
        u += amplitude * torch.sin(2 * np.pi * freq * x / L + phase)
    return u

# Operator definition: identity mapping (u -> u)
def operator_identity(u):
    return u.clone()

# %% 3. Data Generation --------------------------------------------------------------------------------------
inputs_local = []
outputs_local = []
batch_counter = 0

for i in range(rank, N_samples, size):
    if verbose and (i % batch_size == 0):
        print(f"[Rank {rank}] Generating sample {i+1}/{N_samples}")

    u0 = sample_highfreq_signal()
    u1 = operator_identity(u0)

    inputs_local.append(u0.cpu().numpy())
    outputs_local.append(u1.cpu().numpy())

    if len(inputs_local) >= batch_size:
        np.savez_compressed(
            os.path.join(save_dir, f"identity_batch_rank{rank}_{batch_counter}.npz"),
            inputs=np.stack(inputs_local),
            outputs=np.stack(outputs_local),
        )
        inputs_local.clear()
        outputs_local.clear()
        batch_counter += 1

if len(inputs_local) > 0:
    np.savez_compressed(
        os.path.join(save_dir, f"identity_batch_rank{rank}_{batch_counter}.npz"),
        inputs=np.stack(inputs_local),
        outputs=np.stack(outputs_local),
    )
    batch_counter += 1

if verbose:
    print(
        f"✅ Rank {rank}: Data generation complete. "
        f"Saved {batch_counter} batch files "
        f"({batch_counter * batch_size} samples) "
        f"to {save_dir}/identity_batch_rank{rank}_*.npz"
    )

