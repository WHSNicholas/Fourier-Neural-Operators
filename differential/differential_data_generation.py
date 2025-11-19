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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GSTools import for fast Matérn SRF
import gstools as gs

# Parallel Computing
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Parameters - CHANGED TO USE EVEN GRID SIZE
N = 2 ** 6  # Bandlimit Index
n = 512  # Grid Size - NOW EVEN (256)
batch_size = 10  # Batch Size
N_samples = 12 * batch_size * 20 # Number of datapoints generated
verbose = True  # Verbosity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(BASE_DIR, "differential_batches2")
os.makedirs(out_dir, exist_ok=True)
if verbose and rank == 0:
    print(f"[All ranks] Saving batches to: {out_dir}")
    print(f"[All ranks] Using grid size: {n} (even for GSTools compatibility)")

# Reproducibility
seed = 2025
rng = np.random.default_rng(seed + rank)


def _save_pair_image(f2d: np.ndarray, t2d: np.ndarray, filepath: str, title_prefix: str = ""):
    """Save a side-by-side image of input f(x,y) and target to disk."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    im0 = axs[0].imshow(f2d, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    axs[0].set_title(f"{title_prefix} f(x,y)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(t2d, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    axs[1].set_title(f"{title_prefix} ∂x f + ∂y f")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("x");
        ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# %% 2. Generating Function ----------------------------------------------------------------------------------
# Local Counts
local_count = N_samples // size
if rank < N_samples % size:
    local_count += 1

# Containers
inputs_local = []
outputs_local = []
batch_counter = 0

# %% 3. Data Generation Setup --------------------------------------------------------------------------------
# Unshifted frequency multipliers for derivatives (to pair with np.fft.fft2)
freq_1d_un = np.fft.fftfreq(n, d=1.0 / n)  # also integers
KX, KY = np.meshgrid(freq_1d_un, freq_1d_un, indexing="xy")  # unshifted

i2piKX = 1j * 2.0 * np.pi * KX
i2piKY = 1j * 2.0 * np.pi * KY

# Regular [0,1]^2 grid to evaluate GRF directly (no interpolation needed)
grid1d = np.linspace(0.0, 1.0, n, endpoint=False)

# CRITICAL CHANGES: Parameters for high-frequency content
# To get high-frequency content, we need:
# 1. Smaller length scale -> more high-frequency variations
# 2. Lower smoothness parameter (nu) -> rougher field

# High-frequency Matérn field parameters
len_scale_hf = 0.05  # Much smaller length scale for high frequencies
nu_hf = 1.5  # Lower nu for rougher field (more high-frequency content)
var_hf = 1.0  # Variance

# Build the high-frequency Matérn SRF - NOW WITH EVEN mode_no
cov_hf = gs.Matern(dim=2, var=var_hf, len_scale=len_scale_hf, nu=nu_hf)
srf_hf = gs.SRF(cov_hf, generator="Fourier", period=[1.0, 1.0], mode_no=[N, N])


def generate_high_frequency_field(srf, rng, grid1d, n, method='matern_hf'):
    """
    Generate field with high-frequency content.
    """
    if method == 'matern_hf':
        # CRITICAL: Create new SRF with different seed for each sample
        seed = rng.integers(0, 2 ** 31 - 1)
        cov_hf = gs.Matern(dim=2, var=var_hf, len_scale=len_scale_hf, nu=nu_hf)
        srf_new = gs.SRF(cov_hf, generator="Fourier", period=[1.0, 1.0], mode_no=[2*N, 2*N], seed=seed)
        field = srf_new.structured([grid1d, grid1d])

    elif method == 'multi_scale':
        # Superposition of fields at different scales
        field = np.zeros((n, n))
        scales = [0.02, 0.05, 0.1]
        weights = [0.6, 0.3, 0.1]

        for scale, weight in zip(scales, weights):
            seed = rng.integers(0, 2 ** 31 - 1)
            cov_temp = gs.Matern(dim=2, var=weight, len_scale=scale, nu=0.5)
            srf_temp = gs.SRF(cov_temp, generator="Vector", seed=seed)
            field_temp = srf_temp.structured([grid1d, grid1d])
            field += field_temp

    elif method == 'power_law':
        # [Keep existing power_law implementation...]
        # Generate field with power-law spectrum
        kx, ky = np.meshgrid(freq_1d_un, freq_1d_un, indexing='ij')
        k_mag = np.sqrt(kx ** 2 + ky ** 2)
        k_mag[0, 0] = 1.0

        alpha = 2.0
        power_spectrum = k_mag ** (-alpha / 2)
        power_spectrum[0, 0] = 0.0

        # Use rng for random phases
        phases = rng.uniform(0, 2 * np.pi, (n, n))
        complex_amplitudes = power_spectrum * np.exp(1j * phases)

        # Ensure Hermitian symmetry
        complex_amplitudes = np.fft.fftshift(complex_amplitudes)
        center = n // 2
        complex_amplitudes[center + 1:, :] = np.conj(complex_amplitudes[1:center + 1, :][::-1, ::-1])
        complex_amplitudes = np.fft.ifftshift(complex_amplitudes)

        field = np.fft.ifft2(complex_amplitudes).real
        field = field / np.std(field)

    else:
        raise ValueError(f"Unknown method: {method}")

    return field.astype(np.float32)


# %% 4. Data generation --------------------------------------------------------------------------------------
start_time = time.time()
inputs_local = []
outputs_local = []
batch_counter = 0

# Choose generation method
generation_method = 'matern_hf'  # Options: 'matern_hf', 'multi_scale', 'power_law'

# --- Visualisation (rank 0 only) ---
viz_enabled = (rank == 0)
viz_max = 6  # save up to 6 previews
viz_count = 0

for j in range(local_count):
    if verbose and j % batch_size == 0:
        print(f"[Rank {rank}] Generating sample {j + 1}/{local_count} using method: {generation_method}")

    # Generate high-frequency field
    t0 = time.time()
    field = generate_high_frequency_field(srf_hf, rng, grid1d, n, method=generation_method)
    f = field.astype(np.float32)
    t1 = time.time()

    # Spectral gradients:  ∂̂x f = i 2π kx  f̂,  ∂̂y f = i 2π ky f̂
    F_hat = np.fft.fft2(f)
    dfx = np.fft.ifft2(i2piKX * F_hat).real.astype(np.float32)
    dfy = np.fft.ifft2(i2piKY * F_hat).real.astype(np.float32)
    t2 = time.time()

    dsum = (dfx + dfy).astype(np.float32)  # (n, n)

    # Per-sample normalisation (zero mean, unit variance)
    f_mean, f_std = np.mean(f), np.std(f)
    f = (f - f_mean) / (f_std + 1e-8)

    dsum_mean, dsum_std = np.mean(dsum), np.std(dsum)
    dsum = (dsum - dsum_mean) / (dsum_std + 1e-8)

    if verbose and (j % batch_size == 0) and (rank == 0):
        print(f"[Rank {rank}] sample {j + 1}: field={t1 - t0:.3f}s | deriv={t2 - t1:.3f}s")
        print(
            f"[Rank {rank}] Field stats: mean={np.mean(f):.3f}, std={np.std(f):.3f}, min={np.min(f):.3f}, max={np.max(f):.3f}")

    if viz_enabled and viz_count < viz_max:
        preview_path = os.path.join(out_dir, f"preview_rank{rank}_sample{viz_count}_{generation_method}.png")
        _save_pair_image(f, dsum, preview_path, title_prefix=f"j={j}")
        viz_count += 1

    # Collect
    inputs_local.append(f)
    outputs_local.append(dsum)

    # Flush a batch
    if len(inputs_local) >= batch_size:
        np.savez(
            os.path.join(out_dir, f"differential2d_batch_rank{rank}_{batch_counter}_{generation_method}.npz"),
            inputs=np.stack(inputs_local),  # (B, n, n)
            outputs=np.stack(outputs_local),  # (B, n, n)
            method=generation_method,
            parameters={'len_scale': len_scale_hf, 'nu': nu_hf, 'var': var_hf}
        )
        inputs_local.clear()
        outputs_local.clear()
        batch_counter += 1

# Save leftovers
if inputs_local:
    np.savez(
        os.path.join(out_dir, f"differential2d_batch_rank{rank}_{batch_counter}_{generation_method}.npz"),
        inputs=np.stack(inputs_local),
        outputs=np.stack(outputs_local),
        method=generation_method,
        parameters={'len_scale': len_scale_hf, 'nu': nu_hf, 'var': var_hf}
    )
    batch_counter += 1

elapsed = time.time() - start_time
if verbose:
    print(f"✅ Rank {rank}: 2D differential data done in {elapsed:.2f}s. "
          f"Saved {batch_counter} batches ({local_count} samples) to {out_dir}/")
    print(f"✅ Generation method: {generation_method}")
    print(f"✅ High-frequency parameters: len_scale={len_scale_hf}, nu={nu_hf}")