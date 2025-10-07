# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                            2D Darcy Equation - Data Merge                                            #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import glob
import h5py

# Parameters
verbose = True

# %% 2. Merging Data -----------------------------------------------------------------------------------------
# Sample Files
file_paths = sorted(glob.glob("darcy_batch_*.npz"))

# Determine shapes
sample0 = np.load(file_paths[0])
batch0 = sample0["inputs"]
batch_size, C, H, W = batch0.shape
total_samples = batch_size * len(file_paths)

# Write incrementally to HDF5
with h5py.File("darcy_dataset.h5", "w") as f:
    dset_in = f.create_dataset(
        "inputs",
        shape=(total_samples, C, H, W),
        dtype=batch0.dtype,
        chunks=(batch_size, C, H, W)
    )
    # outputs are single‐channel (H, W)
    sample_out = sample0 = sample0 = None  # workaround to load outputs dtype
    sample_out = np.load(file_paths[0])["outputs"]
    dset_out = f.create_dataset(
        "outputs",
        shape=(total_samples, H, W),
        dtype=sample_out.dtype,
        chunks=(batch_size, H, W)
    )

    idx = 0
    for path in file_paths:
        data = np.load(path)
        B = data["inputs"]  # (batch_size, 2, H, W)
        C = data["outputs"]  # (batch_size, H, W)
        dset_in[idx: idx + batch_size] = B
        dset_out[idx: idx + batch_size] = C
        idx += batch_size

if verbose:
    print(f"✅ Merged {len(file_paths)} files into darcy_dataset.h5")
    print(f"Total samples: {total_samples}, each input of shape (2, {H}, {W}) and output of shape ({H}, {W})")