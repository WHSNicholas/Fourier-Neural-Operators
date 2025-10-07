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

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Parameters
d_res = 2**10                      # Domain Resolution
t_res = 2**10                      # Time Resolution
L = 1                              # Domain Length
T = 1                              # Final Time
dt = T / t_res                     # Time Increments
dx = L / d_res                     # Grid Increments
nu = 0                             # Viscosity

batch_size = 120                   # Batch Size
N_samples = batch_size*20          # Number of input/output datapoints generated
verbose = True                     # Verbosity


# %% 2. Pseudospectral Setup ---------------------------------------------------------------------------------
# Defining Mesh and Function Space
x = torch.linspace(0, L, d_res, device=device)
k = 2 * np.pi * torch.fft.rfftfreq(d_res, d=dx).to(device)

# %% 3. Data Generation --------------------------------------------------------------------------------------
inputs = []
outputs = []

for i in range(N_samples):
    # Sample Initial Condition
    u0_expr = sample_field_1d(res=d_res, L=L, type='matern',
                              smoothness=np.sqrt(3)/5, amplitude=1, nu=1.5, var=1.25)
    u0 = torch.from_numpy(u0_expr((x.cpu().numpy(), ))).to(device)
    u1 = u0.clone()

    exp_diff = torch.exp(-nu * k ** 2 * dt)

    for _ in range(t_res):
        # 1. Diffusion
        u_hat = torch.fft.rfft(u1)
        u1 = torch.fft.irfft(u_hat * exp_diff, n=d_res)

        # 2. Convection
        u_hat = torch.fft.rfft(u1)
        u_sq_hat = torch.fft.rfft(0.5 * u1 ** 2)

        N_cutoff = u_sq_hat.shape[-1] * 2 // 3  # Dealiasing
        u_sq_hat[N_cutoff:] = 0

        u_hat = u_hat - dt * (1j * k) * u_sq_hat
        u1 = torch.fft.irfft(u_hat, n=d_res)

    inputs.append(u0.cpu().numpy())
    outputs.append(u1.cpu().numpy())

    if (i + 1) % batch_size == 0:
        batch_id = (i + 1) // batch_size
        np.savez_compressed(f"burgers_batch_{batch_id}.npz",
                            inputs=np.stack(inputs),
                            outputs=np.stack(outputs))
        inputs.clear()
        outputs.clear()

if verbose:
    print(
        f"✅ Data generation complete. "
        f"Saved 20 batch files "
        f"({20 * batch_size} samples) "
        f"as burgers_batch_{batch_id}.npz"
    )

# # Convert to NumPy for plotting
# x_np = x.cpu().numpy()
# u0_np = u0.cpu().numpy()
# u_np = u.cpu().numpy()
# u1_np = u1.cpu().numpy()
#
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))
# axes[0].plot(x_np, u0_np, lw=2)
# axes[0].set_title('Initial Condition $u_0(x)$')
# axes[0].grid(True)
#
# axes[1].plot(x_np, u_np, lw=2, color='orange')
# axes[1].set_title('Solution at $t = 1$ GPT')
# axes[1].grid(True)
#
# axes[2].plot(x_np, u1_np, lw=2, color='orange')
# axes[2].set_title('Solution at $t = 1$ DS')
# axes[2].grid(True)
#
#
# plt.tight_layout()
# plt.show()
#

