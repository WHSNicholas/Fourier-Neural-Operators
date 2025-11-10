# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                      Differential Operator - Data Generation                                      #
# -------------------------------------------------------------------------------------------------------------------- #
# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dependencies
import numpy as np
import math
from mpi4py import MPI
import time

# Parallel Computing
comm = MPI.COMM_WORLD
self = MPI.COMM_SELF
rank = comm.rank
size = comm.size

# Parameters
N = 2**8                           # Bandlimit Index
n = 2 * N + 1                      # Grid Size
p = 1.5                            # Exponent parameter
batch_size = 10                    # Batch Size
N_samples = 12*batch_size*20       # Number of datapoints generated
verbose = True                     # Verbosity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(BASE_DIR, "differential_batches")
os.makedirs(out_dir, exist_ok=True)
if verbose:
    print(f"[Rank {rank}] Saving batches to: {out_dir}")

# Reproducibility
seed = 2025
rng = np.random.default_rng(seed + rank)


# %% 2. Generating Function ----------------------------------------------------------------------------------
# Local Counts
local_count = N_samples // size
if rank < N_samples % size:
    local_count += 1

# Containers
inputs_local = []
outputs_local = []
batch_counter = 0

# %% 3. Data Generation --------------------------------------------------------------------------------------
# Frequencies
k = np.fft.fftfreq(n, d=1.0/n) * n  # integer frequencies: [-N,...,N]
k = np.fft.fftshift(k)              # shift to symmetric order

# Random amplitude envelope
amp = np.zeros_like(k, dtype=float)
mask = k != 0
amp[mask] = 1.0 / ((1.0 + np.abs(k[mask])) * (np.log(math.e + np.abs(k[mask])) ** p))

# %% 4. Data generation --------------------------------------------------------------------------------------
start_time = time.time()

for j in range(local_count):
    if verbose and j % batch_size == 0:
        print(f"[Rank {rank}] Generating sample {j+1}/{local_count}")

    # Random phases
    phases_centered = rng.uniform(0.0, 2.0 * np.pi, size=n)
    C_centered = amp * np.exp(1j * phases_centered)
    C_centered[k == 0.0] = 0.0

    # Enforce Hermitian symmetry for real signal
    C_centered = 0.5 * (C_centered + np.conj(C_centered[::-1]))

    # Inverse FFT -> real signal f(x)
    f = np.fft.ifft(np.fft.ifftshift(C_centered))
    f = np.real(f)
    f_std = f.std()
    if f_std > 0:
        f = f / f_std

    # Derivative in Fourier space:  f_hat' = i * 2π * k * f_hat
    f_hat = np.fft.fft(f)
    df_hat = (1j * 2.0 * np.pi * np.fft.fftfreq(n, d=1.0/n)) * f_hat
    df = np.real(np.fft.ifft(df_hat))

    inputs_local.append(f.astype(np.float32))
    outputs_local.append(df.astype(np.float32))

    if len(inputs_local) >= batch_size:
        np.savez_compressed(
            os.path.join(out_dir, f"differential_batch_rank{rank}_{batch_counter}.npz"),
            inputs=np.stack(inputs_local),
            outputs=np.stack(outputs_local),
        )
        inputs_local.clear()
        outputs_local.clear()
        batch_counter += 1

# Save leftovers
if len(inputs_local) > 0:
    np.savez_compressed(
        os.path.join(out_dir, f"differential_batch_rank{rank}_{batch_counter}.npz"),
        inputs=np.stack(inputs_local),
        outputs=np.stack(outputs_local),
    )
    batch_counter += 1

elapsed = time.time() - start_time
if verbose:
    print(
        f"✅ Rank {rank}: Differential data generation complete in {elapsed:.2f}s. "
        f"Saved {batch_counter} batches ({local_count} samples) to {out_dir}/"
    )