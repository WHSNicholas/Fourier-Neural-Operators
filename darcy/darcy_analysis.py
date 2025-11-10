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
def fno_size(epsilon, d=2, k=1):
    """
    Compute the theoretical FNO network size S for a given error tolerance ε.
    Bound: S = C * ε^{-d/k} * log(1/ε)  (here we return the shape factor without C)
    """
    epsilon = np.asarray(epsilon, dtype=float)
    return epsilon ** (-d / k) * np.log(1.0 / epsilon)

# Data
params = np.array([11241, 66113, 533633, 2029249, 2702353], dtype=float)
errors = np.array([ 0.3583, 0.0738, 0.0213, 0.0115, 0.0166], dtype=float)

# Fit multiplicative constant C in log-space so theory overlays empirical S(ε)
f_eps = fno_size(errors, d=2, k=1)
C = np.exp(np.mean(np.log(params) - np.log(f_eps)))  # best-fit scalar

# Build a smooth theory curve on the same axes (x: params S, y: error ε)
eps_min, eps_max = errors.min(), errors.max()
eps_curve = np.logspace(np.log10(eps_min * 0.5), np.log10(eps_max * 1.1), 400)
S_curve = C * fno_size(eps_curve, d=2, k=1)

# Plotting
plt.figure(figsize=(8, 5))

# Theory as a smooth curve
plt.plot(S_curve, eps_curve, color='red', linewidth=2, label=f'Theory $(d=2, k=1)$')

# Empirical points (scatter) and a light connector for readability
plt.scatter(params, errors, s=25, color='blue', label='Empirical')
plt.plot(params, errors, color='blue', alpha=0.5, linestyle='--')

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Number of Parameters')
plt.ylabel(r'Test $L^2$ Error')
plt.title(r'Test $L^2$ Error vs Model Size')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(-0.02, eps_max * 1.2)
plt.legend()

plt.tight_layout()
plt.savefig('figures/dar_eff_app.png', dpi=300)
plt.show()



# %% 4. Resolution Invariance --------------------------------------------------------------------------------
# Data
res = np.array([16, 32, 64, 128, 256, 512], dtype=float)
errors_L2 = np.array([0.0585, 0.06, 0.0768, 0.0805, 0.0738, 0.074], dtype=float)
errors_Linf = np.array([0.0727, 0.0956, 0.1607, 0.2089, 0.2319, 0.2602], dtype=float)
errors_H1 = np.array([0.1322, 0.2231, 0.3878, 0.5621, 0.6504, 0.7177], dtype=float)

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
plt.savefig('figures/dar_res_inv.png', dpi=300)
plt.show()

# %% 5. Zero-Shot Super-Resolution ---------------------------------------------------------------------------
# 64 Grid Super Resolution
# Data
res64 = np.array([64, 128, 256, 512], dtype=float)
L2_64 = np.array([0.0768, 0.0752, 0.0816, 0.0804], dtype=float)
Linf_64 = np.array([0.1607, 0.2056, 0.2575, 0.2627], dtype=float)
H1_64 = np.array([0.3878, 0.5247, 0.6665, 0.7588], dtype=float)

# 16 Grid Super Resolution
# Data
res16 = np.array([16, 32, 64, 128, 256, 512], dtype=float)
L2_16 = np.array([0.0531, 0.2153, 0.237, 0.2601, 0.2655, 0.2691], dtype=float)
Linf_16 = np.array([0.064, 0.2817, 0.3765, 0.4451, 0.5195, 0.5397], dtype=float)
H1_16 = np.array([0.1154, 0.4931, 0.7352, 0.8749, 0.9937, 1.0631], dtype=float)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel
ax = axes[0]
order64 = np.argsort(res64)
ax.scatter(res64, L2_64, s=25, color='blue', label=r'$L^2$ Error')
ax.plot(res64[order64], L2_64[order64], color='blue', alpha=0.5, linestyle='--')

ax.scatter(res64, Linf_64, s=25, color='red', label=r'$L^\infty$ Error')
ax.plot(res64[order64], Linf_64[order64], color='red', alpha=0.5, linestyle='--')

ax.scatter(res64, H1_64, s=25, color='green', label=r'$H^1$ Error')
ax.plot(res64[order64], H1_64[order64], color='green', alpha=0.5, linestyle='--')

# ax.set_xscale('log', base=2)
ax.set_xlabel('Mesh Resolution')
ax.set_ylabel(r'Test Error')
ax.set_title(r'Zero-Shot Super-Resolution trained at $64 \times 64$')
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
ax.set_title(r'Zero-Shot Super-Resolution trained at $16 \times 16$')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
plt.savefig('figures/dar_zssr.png', dpi=300)
plt.show()