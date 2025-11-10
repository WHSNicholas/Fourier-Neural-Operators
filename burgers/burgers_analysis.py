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

# %% 2. Efficient Approximation ------------------------------------------------------------------------------
_use_cmu_tex()
def fno_size(epsilon, d=2, k=1):
    """
    Compute the theoretical FNO network size S for a given error tolerance ε.
    Bound: S = C * ε^{-d/k} * log(1/ε)  (here we return the shape factor without C)
    """
    epsilon = np.asarray(epsilon, dtype=float)
    return epsilon ** (-d / k) * np.log(1.0 / epsilon)

# Data
params = np.array([485, 2249, 12289, 60417, 162817, 287569], dtype=float)
errors = np.array([0.0217, 0.0072, 0.0041, 0.0123, 0.0085, 0.0053], dtype=float)

# Fit multiplicative constant C in log-space so theory overlays empirical S(ε)
f_eps = fno_size(errors, d=1, k=0.5)
C = np.exp(np.mean(np.log(params) - np.log(f_eps)))  # best-fit scalar

# Build a smooth theory curve on the same axes (x: params S, y: error ε)
eps_min, eps_max = errors.min(), errors.max()
eps_curve = np.logspace(np.log10(eps_min * 0.2), np.log10(eps_max * 1.1), 400)
S_curve = C * fno_size(eps_curve, d=1, k=0.5)

# Plotting
plt.figure(figsize=(8, 5))

# Theory as a smooth curve
plt.plot(S_curve, eps_curve, color='red', linewidth=2, label=f'Theory $(d=1, k=1)$')

# Empirical points (scatter) and a light connector for readability
plt.scatter(params, errors, s=25, color='blue', label='Empirical')
plt.plot(params, errors, color='blue', alpha=0.5, linestyle='--')

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Number of Parameters')
plt.ylabel(r'Test $L^2$ Error')
plt.title(r'Test $L^2$ Error vs Model Size')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(-0.0, eps_max * 1.2)
plt.legend()

plt.tight_layout()
plt.savefig('efficient_approx_burgers.png', dpi=300)


# %% 4. Resolution Invariance --------------------------------------------------------------------------------
# Data
res = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
errors_L2 = np.array([0.013, 0.0051, 0.0046, 0.0048, 0.0054, 0.0044, 0.0056, 0.0123, 0.0069], dtype=float)
errors_Linf = np.array([0.0166, 0.007, 0.007, 0.0074, 0.0078, 0.0075, 0.0084, 0.0178, 0.0101], dtype=float)
errors_H1 = np.array([0.0246, 0.0219, 0.0333, 0.0413, 0.044, 0.0483, 0.0445, 0.0455, 0.0685], dtype=float)

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
plt.savefig('res_invariance_burgers.png', dpi=300)

# %% 5. Zero-Shot Super-Resolution ---------------------------------------------------------------------------
# 64 Grid Super Resolution
# Data
res = np.array([256, 512, 1024, 2048, 4096], dtype=float)
errors_L2 = np.array([0.0054, 0.0045, 0.0052, 0.0053, 0.0051], dtype=float)
errors_Linf = np.array([0.0075, 0.0061, 0.007, 0.0071, 0.0069], dtype=float)
errors_H1 = np.array([0.0483, 0.0361, 0.0442, 0.0496, 0.0653], dtype=float)

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
plt.title(r'Zero-Shot Super-Resolution trained at $256$')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig('zs_sup_res_burgers_256.png', dpi=300)

# 16 Grid Super Resolution
# Data
res = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
errors_L2 = np.array([0.005, 0.0076, 0.008, 0.0081, 0.0094, 0.0075, 0.008, 0.0076, 0.0073], dtype=float)
errors_Linf = np.array([0.007, 0.0121, 0.0145, 0.0141, 0.0163, 0.0135, 0.0142, 0.0138, 0.0128], dtype=float)
errors_H1 = np.array([0.0146, 0.0564, 0.1012, 0.1183, 0.1358, 0.1352, 0.1413, 0.1642, 0.1765], dtype=float)


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
plt.title(r'Zero-Shot Super-Resolution trained at $16$')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig('zs_sup_res_burgers_16.png', dpi=300)
