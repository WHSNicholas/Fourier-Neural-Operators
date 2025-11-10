# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                          1D Identity Operator - Data Merge                                           #
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
file_paths = sorted(glob.glob("identity/identity_batches/identity_batch_*.npz"))

# Determine Shapes
sample0 = np.load(file_paths[0])
batch0 = sample0["inputs"]
sample_out = sample0["outputs"]

batch_size, N_nodes = batch0.shape     # Input Shape
out_shape = sample_out.shape           # Output Shape

if len(out_shape) == 3:
    T = out_shape[1]
elif len(out_shape) == 2:
    T = 1
else:
    raise ValueError(f"Unexpected output shape: {out_shape}")

total_samples = batch_size * len(file_paths)

# Write incrementally to HDF5
with h5py.File("identity_dataset.h5", "w") as f:
    # Inputs
    dset_in = f.create_dataset(
        "inputs",
        shape=(total_samples, N_nodes),
        dtype=batch0.dtype,
        chunks=(batch_size, N_nodes)
    )

    # Outputs
    dset_out = f.create_dataset(
        "outputs",
        shape=(total_samples, T, N_nodes),
        dtype=sample_out.dtype,
        chunks=(batch_size, T, N_nodes)
    )

    idx = 0
    for path in file_paths:
        data = np.load(path)
        B = data["inputs"]
        C = data["outputs"]

        if C.ndim == 2:
            C = C[:, None, :]

        dset_in[idx : idx + batch_size] = B
        dset_out[idx : idx + batch_size] = C
        idx += batch_size

print(f"✅ Merged {len(file_paths)} files into identity_dataset.h5")
print(f"Total samples: {total_samples}, each input length {N_nodes}, output shape ({T}, {N_nodes})")


# # -------------------------------------------------------------------------------------------------------------------- #
# #                                               Fourier Neural Operators                                               #
# #                                           2D Poisson Equation - Data Merge                                           #
# # -------------------------------------------------------------------------------------------------------------------- #
#
# # %% 1. Preamble ---------------------------------------------------------------------------------------------
# import numpy as np
# import glob
# import h5py
#
# # Parameters
# verbose = True
#
# # %% 2. Merging Data -----------------------------------------------------------------------------------------
# # Sample Files
# file_paths = sorted(glob.glob("highfreq/highfreq_batches/highfreq_batch_*.npz"))
#
# # Determine shapes
# sample0 = np.load(file_paths[0])
# batch0 = sample0["inputs"]
# batch_size, H, W = batch0.shape
# total_samples = batch_size * len(file_paths)
#
# # Write incrementally to HDF5
# with h5py.File("highfreq_dataset.h5", "w") as f:
#     dset_in = f.create_dataset("inputs", shape=(total_samples, H, W),
#                                dtype=batch0.dtype, chunks=(batch_size, H, W))
#     dset_out = f.create_dataset("outputs", shape=(total_samples, H, W),
#                                 dtype=sample0["outputs"].dtype,
#                                 chunks=(batch_size, H, W))
#
#     idx = 0
#     for path in file_paths:
#         data = np.load(path)
#         B = data["inputs"]       # (batch_size, H, W)
#         C = data["outputs"]
#         dset_in[idx: idx + batch_size] = B
#         dset_out[idx: idx + batch_size] = C
#         idx += batch_size
#
# if verbose:
#     print(f"✅ Merged {len(file_paths)} files into highfreq_dataset.h5")
#     print(f"Total samples: {total_samples}, each of shape ({H},{W})")