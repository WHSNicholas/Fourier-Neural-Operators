# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                             2D Darcy Equation - Analysis                                             #
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
def fno_size(epsilon, d=1, k=0.5):
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

# Data
params = np.array([485, 2249, 12289, 60417, 162817, 287569], dtype=float)
errors = np.array([0.0098, 0.0094, 0.0054, 0.0048, 0.004, 0.004], dtype=float)


# Fit multiplicative constant C in log-space so theory overlays empirical S(ε)
f_eps = fno_size(errors)
f_eps2 = fno_size(errors, k = 3)
C = np.exp(np.mean(np.log(params) - np.log(f_eps)))
C2 = np.exp(np.mean(np.log(params) - np.log(f_eps2)))

# Build a smooth theory curve on the same axes (x: params S, y: error ε)
eps_min, eps_max = errors.min(), errors.max()
eps_curve = np.logspace(np.log10(eps_min * 0.1), np.log10(eps_max * 2), 400)
S_curve = C * fno_size(eps_curve)
S_curve2 = C2 * fno_size(eps_curve, k = 3)

# Plotting
plt.figure(figsize=(8, 5))

# Theory as a smooth curve
plt.plot(S_curve, eps_curve, color='red', linewidth=2, label=f'Theory $(d=1, k=0.5)$')
plt.plot(S_curve2, eps_curve, color='green', linewidth=2, label=f'Theory $(d=1, k=3)$')

# Empirical points (scatter) and a light connector for readability
plt.scatter(params, errors, s=25, color='blue', label='Empirical')
plt.plot(params, errors, color='blue', alpha=0.5, linestyle='--')

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Number of Parameters')
plt.ylabel(r'Test $L^2$ Error')
plt.title(r'Test $L^2$ Error vs Model Size')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(-0.0, eps_max * 2)
plt.xlim(0.8*params[0], 1.2*params[-1])
plt.legend()

plt.tight_layout()
plt.savefig('figures/bur_eff_app.png', dpi=300)


# %% 4. Resolution Invariance --------------------------------------------------------------------------------
# Data
res = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
errors_L2 = np.array([0.005, 0.0036, 0.0059, 0.0038, 0.0041, 0.0118, 0.006, 0.0048, 0.0052], dtype=float)
errors_Linf = np.array([0.0072, 0.0055, 0.0076, 0.0064, 0.0072, 0.0155, 0.01, 0.0088, 0.0094], dtype=float)
errors_H1 = np.array([0.0143, 0.0249, 0.0358, 0.0542, 0.0686, 0.0513, 0.1128, 0.1195, 0.11269], dtype=float)


# Plotting
plt.figure(figsize=(8, 5))
order = np.argsort(res)
plt.scatter(res, errors_L2, s=25, color='blue', label=r'$L^2$ Error')
plt.plot(res[order], errors_L2[order], color='blue', alpha=0.5, linestyle='--')

plt.scatter(res, errors_Linf, s=25, color='red', label=r'$L^\infty$ Error')
plt.plot(res[order], errors_Linf[order], color='red', alpha=0.5, linestyle='--')

plt.scatter(res, errors_H1, s=25, color='green', label=r'$H^1$ Error')
plt.plot(res[order], errors_H1[order], color='green', alpha=0.5, linestyle='--')

#plt.xscale('log', base=2)

plt.xlabel('Mesh Resolution')
plt.ylabel(r'Test Error')
plt.title(r'Resolution Invariance')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig('figures/bur_res_inv.png', dpi=300)

# %% 5. Zero-Shot Super-Resolution ---------------------------------------------------------------------------
# 64 Grid Super Resolution
# Data
res256 = np.array([256, 512, 1024, 2048, 4096], dtype=float)
L2_256 = np.array([0.0043, 0.0041, 0.0043, 0.0041, 0.0041], dtype=float)
Linf_256 = np.array([0.0075, 0.0071, 0.0075, 0.0072, 0.0071], dtype=float)
H1_256 = np.array([0.0708, 0.0728, 0.0769, 0.085, 0.0995], dtype=float)

# 16 Grid Super Resolution
# Data
res16 = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
L2_16 = np.array([0.0052, 0.0087, 0.0079, 0.0079, 0.0075, 0.0076, 0.0078, 0.0081, 0.0081], dtype=float)
Linf_16 = np.array([0.0077, 0.0138, 0.0135, 0.0138, 0.0135, 0.0138, 0.014, 0.0146, 0.0146], dtype=float)
H1_16 = np.array([0.0146, 0.0777, 0.1263, 0.1437, 0.1528, 0.1732, 0.1702, 0.2134, 0.2716], dtype=float)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel
ax = axes[0]
order256 = np.argsort(res256)
ax.scatter(res256, L2_256, s=25, color='blue', label=r'$L^2$ Error')
ax.plot(res256[order256], L2_256[order256], color='blue', alpha=0.5, linestyle='--')

ax.scatter(res256, Linf_256, s=25, color='red', label=r'$L^\infty$ Error')
ax.plot(res256[order256], Linf_256[order256], color='red', alpha=0.5, linestyle='--')

ax.scatter(res256, H1_256, s=25, color='green', label=r'$H^1$ Error')
ax.plot(res256[order256], H1_256[order256], color='green', alpha=0.5, linestyle='--')

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
plt.savefig('figures/bur_zssr.png', dpi=300)
plt.show()