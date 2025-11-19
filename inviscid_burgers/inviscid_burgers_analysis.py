# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                             1D Burgers Equation - Analysis                                             #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils.analysis import _use_cmu_tex

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})

# %% 2. Efficient Approximation ------------------------------------------------------------------------------
_use_cmu_tex()
def fno_size_ours(epsilon, d=1, k=0.5):
    """
    Shape factor (without the constant C) for the bound:
        S ~ ε^{-(1 + 1/r)} * log(ε)
    Parameters
    ----------
    epsilon : float or array-like
        Target error tolerance(s), must be > 0.
    d : int
        Spatial dimension.
    k : float
        Regularity/smoothness parameter in the estimate.

    Returns
    -------
    np.ndarray or float
        ε^{-(1/2 + d/r)}
    """
    eps = np.asarray(epsilon, dtype=float)
    if np.any(eps <= 0):
        raise ValueError("epsilon must be > 0.")
    exponent = -(1 + d / float(k))

    return - np.power(eps, exponent) * np.log(eps)

def fno_size_lit(epsilon):
    """
    Shape factor (without the constant C) for the bound:
        S ~ log(ε^{-1}) ^ 2
    Parameters
    ----------
    epsilon : float or array-like
        Target error tolerance(s), must be > 0.
    d : int
        Spatial dimension.
    k : float
        Regularity/smoothness parameter in the estimate.

    Returns
    -------
    np.ndarray or float
        ε^{-(1/2 + d/r)}
    """
    eps = np.asarray(epsilon, dtype=float)
    if np.any(eps <= 0):
        raise ValueError("epsilon must be > 0.")

    return np.log(eps ** (-1))**2

# Data
params = np.array([485, 2249, 12289, 60417, 162817, 287569, 389161], dtype=float)
errors = np.array([0.2504, 0.1343, 0.0697, 0.0552, 0.0403, 0.04, 0.0373], dtype=float)

# Fit multiplicative constant C in log-space so theory overlays empirical S(ε)
f_eps_ours = fno_size_ours(errors)
f_eps_ours2 = fno_size_ours(errors, k=3)
f_eps_lit = fno_size_lit(errors)
C_ours = np.exp(np.mean(np.log(params) - np.log(f_eps_ours)))
C_ours2 = np.exp(np.mean(np.log(params) - np.log(f_eps_ours2)))
C_lit = np.exp(np.mean(np.log(params) - np.log(f_eps_lit)))

# Build a smooth theory curve on the same axes (x: params S, y: error ε)
eps_min, eps_max = errors.min(), errors.max()
eps_curve = np.logspace(np.log10(eps_min * 0.001), np.log10(eps_max * 2), 400)
S_curve_ours = C_ours * fno_size_ours(eps_curve)
S_curve_ours2 = C_ours2 * fno_size_ours(eps_curve, k=3)
S_curve_lit = C_lit * fno_size_lit(eps_curve)

# Plotting
plt.figure(figsize=(8, 5))

# Theory as a smooth curve
plt.plot(S_curve_ours, eps_curve, color='green', linewidth=2, label=f'Viscous Burgers Theory $(d=1, k=0.5)$')
plt.plot(S_curve_ours2, eps_curve, color='orange', linewidth=2, label=f'Viscous Burgers Theory $(d=1, k=3)$')
plt.plot(S_curve_lit, eps_curve, color='red', linewidth=2, label=f'Literature Theory')

# Empirical points (scatter) and a light connector for readability
plt.scatter(params, errors, s=25, color='blue', label='Empirical')
plt.plot(params, errors, color='blue', alpha=0.5, linestyle='--')

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Number of Parameters')
plt.ylabel(r'Test $L^2$ Error')
plt.title(r'Test $L^2$ Error vs Model Size')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(-0.0, eps_max * 1.5)
plt.xlim(0.5*params[0], 1.2*params[-1])
plt.legend()

plt.tight_layout()
plt.savefig('figures/ibur_eff_app.png', dpi=300)


# %% 4. Resolution Invariance --------------------------------------------------------------------------------
# Data
res = np.array([16, 32, 64, 128, 256, 512, 1024, 2048], dtype=float)
errors_L2 = np.array([0.005, 0.0229, 0.0435, 0.0437, 0.0429, 0.0397, 0.0441, 0.0552], dtype=float)
errors_Linf = np.array([0.0072, 0.1022, 0.1396, 0.2426, 0.3229, 0.4880, 0.6820, 0.8600], dtype=float)
errors_H1 = np.array([0.0143, 0.1585, 0.3694, 0.64332, 0.7809, 1.0241, 1.1669, 1.8973], dtype=float)

# Plotting
plt.figure(figsize=(8, 5))
order = np.argsort(res)
plt.scatter(res, errors_L2, s=25, color='blue', label=r'$L^2$ Error')
plt.plot(res[order], errors_L2[order], color='blue', alpha=0.5, linestyle='--')

plt.scatter(res, errors_Linf, s=25, color='red', label=r'$L^\infty$ Error')
plt.plot(res[order], errors_Linf[order], color='red', alpha=0.5, linestyle='--')

plt.scatter(res, errors_H1, s=25, color='green', label=r'$H^1$ Error')
plt.plot(res[order], errors_H1[order], color='green', alpha=0.5, linestyle='--')

plt.xlabel('Mesh Resolution')
plt.ylabel(r'Test Error')
plt.title(r'Resolution Invariance')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()

plt.tight_layout()
# plt.savefig('figures/ibur_res_inv.png', dpi=300)

# %% 5. Zero-Shot Super-Resolution ---------------------------------------------------------------------------
# 64 Grid Super Resolution
# Data
res128 = np.array([128, 256, 512, 1024, 2048], dtype=float)
L2_128 = np.array([0.0422, 0.0989, 0.1121, 0.1098, 0.1007], dtype=float)
Linf_128 = np.array([0.2389, 0.8703, 1.2860, 1.4165, 1.385], dtype=float)
H1_128 = np.array([0.5929, 1.5706, 2.3019, 2.5658, 2.8653], dtype=float)

# 16 Grid Super Resolution
# Data
res16 = np.array([16, 32, 64, 128, 256, 512, 1024, 2048], dtype=float)
L2_16 = np.array([0.0321, 0.1586, 0.2138, 0.2435, 0.2131, 0.2186, 0.2257, 0.2469], dtype=float)
Linf_16 = np.array([0.0738, 0.5417, 0.9910, 1.4792, 1.5323, 1.7977, 1.7695, 2.0458], dtype=float)
H1_16 = np.array([0.1426, 0.8097, 1.2744, 2.0320, 2.5946, 3.1244, 3.2199, 3.3466], dtype=float)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel
ax = axes[0]
order256 = np.argsort(res128)
ax.scatter(res128, L2_128, s=25, color='blue', label=r'$L^2$ Error')
ax.plot(res128[order256], L2_128[order256], color='blue', alpha=0.5, linestyle='--')

ax.scatter(res128, Linf_128, s=25, color='red', label=r'$L^\infty$ Error')
ax.plot(res128[order256], Linf_128[order256], color='red', alpha=0.5, linestyle='--')

ax.scatter(res128, H1_128, s=25, color='green', label=r'$H^1$ Error')
ax.plot(res128[order256], H1_128[order256], color='green', alpha=0.5, linestyle='--')

# ax.set_xscale('log', base=2)
ax.set_xlabel('Mesh Resolution')
ax.set_ylabel(r'Test Error')
ax.set_title(r'Zero-Shot Super-Resolution trained at $256$')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()


# Right panel
ax = axes[1]
order16 = np.argsort(res16)
ax.scatter(res16, L2_16, s=25, color='blue', label=r'$L^2$ Error')
ax.plot(res16[order16], L2_16[order16], color='blue', alpha=0.5, linestyle='--')

ax.scatter(res16, Linf_16, s=25, color='red', label=r'$L^\infty$ Error')
ax.plot(res16[order16], Linf_16[order16], color='red', alpha=0.5, linestyle='--')

ax.scatter(res16, H1_16, s=25, color='green', label=r'$H^1$ Error')
ax.plot(res16[order16], H1_16[order16], color='green', alpha=0.5, linestyle='--')

# ax.set_xscale('log', base=2)
ax.set_xlabel('Mesh Resolution')
ax.set_title(r'Zero-Shot Super-Resolution trained at $16$')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
# plt.savefig('figures/ibur_zssr.png', dpi=300)
plt.show()