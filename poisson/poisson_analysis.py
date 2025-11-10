# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                            2D Poisson Equation - Analysis                                            #
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
params = np.array([1069, 11225, 66081, 533569, 2029153, 2702257], dtype=float)
errors = np.array([0.1989, 0.0346, 0.0155, 0.0083, 0.0077, 0.0041], dtype=float)

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
order = np.argsort(errors)
plt.scatter(params, errors, s=25, color='blue', label='Empirical')
plt.plot(params[order], errors[order], color='blue', alpha=0.5, linestyle='--')

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Number of Parameters')
plt.ylabel(r'Test $L^2$ Error')
plt.title(r'Test $L^2$ Error vs Model Size')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(-0.02, eps_max * 1.2)
plt.legend()

plt.tight_layout()
plt.savefig('figures/poi_eff_app.png', dpi=300)
plt.show()


# %% 3. Statistical Regression -------------------------------------------------------------------------------
x1 = -np.log(errors)
x2 = np.log(np.log(1/errors))

X = np.vstack([x1, x2]).T         # Design Matrix
X = sm.add_constant(X)

y = np.log(params)
model = sm.OLS(y, X).fit()
print(model.summary())

# Theoretical Tests
d, k = 2, 1
beta1_theory = d/k
beta2_theory = 1.0

# T Tests
test1 = model.t_test([0, 1, 0])
test2 = model.t_test([0, 0, 1])

test1_thry = model.t_test((np.array([0, 1, 0]), beta1_theory))
test2_thry = model.t_test((np.array([0, 0, 1]), beta2_theory))

print("Test for β1 = 0:")
print(test1)
print("\nTest for β2 = 0:")
print(test2)

print("Test for β1 = d/k:")
print(test1_thry)
print("\nTest for β2 = 1:")
print(test2_thry)

# %% 4. Resolution Invariance --------------------------------------------------------------------------------
# Data
res = np.array([16, 32, 64, 128, 256, 512], dtype=float)
errors_L2 = np.array([0.0116, 0.0117, 0.0123, 0.0187, 0.0155, 0.0143], dtype=float)
errors_Linf = np.array([0.0146, 0.0215, 0.0267, 0.0422, 0.0542, 0.0548], dtype=float)
errors_H1 = np.array([0.0235, 0.0413, 0.0643, 0.1124, 0.1473, 0.1513], dtype=float)

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
plt.savefig('figures/poi_res_inv.png', dpi=300)
plt.show()

# %% 5. Zero-Shot Super-Resolution ---------------------------------------------------------------------------
# 64 Grid Super Resolution
# Data
res64 = np.array([64, 128, 256, 512], dtype=float)
L2_64 = np.array([0.0123, 0.0163, 0.0189, 0.0201], dtype=float)
Linf_64 = np.array([0.0267, 0.0429, 0.0588, 0.0659], dtype=float)
H1_64 = np.array([0.0643, 0.1229, 0.1926, 0.2581], dtype=float)

# 16 Grid Super Resolution
# Data
res16 = np.array([16, 32, 64, 128, 256, 512], dtype=float)
L2_16 = np.array([0.0106, 0.0997, 0.1194, 0.1275, 0.1254, 0.1239], dtype=float)
Linf_16 = np.array([0.0133, 0.2070, 0.3514, 0.4384, 0.4773, 0.4857], dtype=float)
H1_16 = np.array([0.0219, 0.3673, 0.7108, 1.0433, 1.4159, 1.8955], dtype=float)

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
plt.savefig('figures/poi_zssr.png', dpi=300)
plt.show()