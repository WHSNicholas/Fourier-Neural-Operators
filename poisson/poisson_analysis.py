# plot_fno_size_vs_error.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data from your table
# -----------------------------
labels   = ["Tiny", "Small", "Medium", "Large", "Huge"]
params   = np.array([1069, 11225, 66081, 533569, 2029153], dtype=float)
err_pct  = np.array([15.68, 4.08, 1.22, 0.83, 0.70], dtype=float)  # in percent

# x-axis will be "percent as decimal" to match ticks 0.00 ... 0.16
x = err_pct / 100.0

# Sort by increasing error for a nice monotone line
order = np.argsort(x)
x = x[order]
y_emp = params[order]
labels = [labels[i] for i in order]

# -----------------------------
# Theory curve: size ~ (1/eps)^(d/k) * log(1/eps)
# Choose k (Darcy: k>=3 in 2D; Poisson example: k=1)
# -----------------------------
d = 2
k = 1.5          # <-- set to 1 if you want the Poisson (f -> u) scaling
eps = x

# avoid issues if any eps == 0 (not the case here)
f_eps = (1.0/eps)**(d/k) * np.log(1.0/eps)

# Fit a single multiplicative constant C in log-space so the curve is on-scale
C = np.exp(np.mean(np.log(y_emp) - np.log(f_eps)))
y_the = C * f_eps

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7.0, 4.2))
plt.plot(x, y_emp, marker='o', linewidth=2, label="Empirical")
plt.plot(x, y_the, marker='o', linewidth=2, label="Theory (d=2, k=%d)" % k)

# annotate points with model names
for xi, yi, lab in zip(x, y_emp, labels):
    plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)

#plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.xlabel(r"Test $L^2$ Error (%)")
plt.ylabel("Number of Parameters (logarithmic)")
plt.title(r"Test $L^2$ Error vs Model Size")
plt.legend()
plt.tight_layout()
plt.show()