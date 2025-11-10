# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                             Utilities - Data Generation                                              #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import gstools as gs
import torch.nn.functional as F
import h5py
import torch

from collections import defaultdict
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset



# %% 2. Objects ----------------------------------------------------------------------------------------------
class Dataset(Dataset):
    def __init__(self, path, res=None):
        self.path = path
        self.h5 = None
        self.res = res

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")

    def __len__(self):
        self._open()
        return self.h5["inputs"].shape[0]

    def __getitem__(self, i):
        self._open()
        inp = self.h5["inputs"][i]
        out = self.h5["outputs"][i]

        # Detect Domain Dimension
        if inp.ndim == 1:
            # 1D Input
            inp = inp[None, ...]
            inp_t = torch.from_numpy(inp).float()
            out_t = torch.from_numpy(out).float()
            if out_t.ndim == 1:
                out_t = out_t.unsqueeze(0)
        else:
            # 2D Input
            if inp.ndim == 2:
                inp = inp[None, ...]
            inp_t = torch.from_numpy(inp).float()
            out_t = torch.from_numpy(out).float().unsqueeze(0)


        # # Ensure channel dimension for single-channel data
        # if inp.ndim == 2:
        #     inp = inp[None, ...]        # (1, H, W)
        #
        # inp_t = torch.from_numpy(inp).float()         # (C, H, W)
        # out_t = torch.from_numpy(out).float().unsqueeze(0)  # (1, H, W)

        # # Downsampling
        # if self.res is not None:
        #     inp_t = F.interpolate(inp_t.unsqueeze(0), size=(self.res, self.res),
        #                           mode='bilinear', align_corners=False)[0]
        #     out_t = F.interpolate(out_t.unsqueeze(0), size=(self.res, self.res),
        #                           mode='bilinear', align_corners=False)[0]

        # Downsampling
        if self.res is not None:
            if inp_t.ndim == 2:  # 1D data
                inp_t = F.interpolate(inp_t.unsqueeze(0), size=self.res, mode='linear', align_corners=False)[0]
                out_t = F.interpolate(out_t.unsqueeze(0), size=self.res, mode='linear', align_corners=False)[0]

            elif inp_t.ndim == 3:                         # 2D Data
                inp_t = F.interpolate(inp_t.unsqueeze(0), size=(self.res, self.res),
                                      mode='bilinear', align_corners=False)[0]
                out_t = F.interpolate(out_t.unsqueeze(0), size=(self.res, self.res),
                                      mode='bilinear', align_corners=False)[0]

        return {"x": inp_t, "y": out_t}



# %% 3. Functions --------------------------------------------------------------------------------------------
def sample_field_1d(
    res: int = 512,
    L: float = 2*np.pi,
    type: str = "fourier",
    modes: int = 10,
    smoothness: float = 0.5,
    amplitude: float = 1.0,
    nu: float = 1.5,
    var: float = 1.25,
):
    """
    Returns f(x) with x[0] an array of 1D coordinates on a periodic domain [0, L).

    Type:
      - "fourier_1d": periodic Fourier series with coefficient decay ~(1+n^2)^(-smoothness/2).
      - "matern_gs_1d": GSTools Matérn on a 1D periodic grid via FFT (SRF(mode='fft')),
                        parameterized to match Cov = τ²(-Δ+κ² I)^(-2) when nu=1.5.
                        (Keep amplitude=1.0 to preserve that covariance exactly.)
    """
    # Fourier-series generator
    def _fourier():
        cov = gs.Gaussian(dim=1, var=1.0, len_scale=smoothness)
        srf = gs.SRF(cov, generator="Fourier", period=[L], mode_no=res)
        grid = np.linspace(0.0, L, res, endpoint=False)
        field = srf.structured([grid])

        def f(x):
            xx = np.asarray(x[0])
            idx = np.floor((np.mod(xx, L) * res / L) + 0.5).astype(int) % res
            return amplitude * field[idx]

        return f

    # Matérn GRF
    def _matern():
        cov = gs.Matern(dim=1, var=var, len_scale=smoothness, nu=nu)
        srf = gs.SRF(cov, generator="Fourier", period=[L], mode_no=res)
        grid = np.linspace(0.0, L, res, endpoint=False)
        field = srf.structured([grid])

        def f(x):
            xx = np.asarray(x[0])
            idx = np.floor((np.mod(xx, L) * res / L) + 0.5).astype(int) % res
            return amplitude * field[idx]
        return f

    samplers = {
        "fourier": _fourier,
        "matern": _matern,
    }

    key = type.lower()
    if key not in samplers:
        raise ValueError(f"Unknown 1D sampler type '{type}'. Use one of {list(samplers.keys())}.")
    return samplers[key]()


def sample_field_2d(
    res: int = 512,
    num_modes_x: int = 10,
    num_modes_y: int = 10,
    degree: int = 5,
    len_scale: float = 0.5,
    var: float = 1.0,
    smoothness: float = 0.5,
    amplitude: float = 1.0,
    type: str = "grf",
):
    """
    Parameters
    ----------
    num_modes_x, num_modes_y : int
        Number of spectral modes (for Fourier) or degrees/density controls for other samplers.
    smoothness : float
        Single smoothness control:
        • Fourier    : spectral decay exponent (higher ⇒ smoother).
        • Polynomial : degree = int(smoothness) if sampler_type='polynomial'.
        • GRBF       : Gaussian bump width σ = 1/smoothness.
        • GRF types  : correlation length ℓ = smoothness.
        • Perlin     : base noise frequency = smoothness.
    amplitude : float
        Global amplitude or standard deviation for all sampler types.
    type : str
        One of: "fourier_1d", "fourier", "polynomial", "grbf", "white_noise", "perlin", "grf", "matern", "exp".

    Returns
    -------
    forcing_func : callable
        A function f(x) that accepts a tuple / array‑dict of coordinates with x[0] = x‑coords, x[1] = y‑coords  (each
        flat), and returns f values at those points. For 'fourier_1d', returns a function of x[0] (1D).
    """
    # Fourier-series generator
    def _fourier():
        n = np.arange(1, num_modes_x + 1)[:, None]
        m = np.arange(1, num_modes_y + 1)[:, None]
        sigma = 1.0 / (1.0 + n**2 + m.T**2) ** (smoothness / 2)
        coeffs = np.random.randn(num_modes_x, num_modes_y) * sigma
        def f(x):
            xx = x[0].reshape(1, -1)
            yy = x[1].reshape(1, -1)
            sinx = np.sin(np.pi * n * xx)
            siny = np.sin(np.pi * m * yy)
            out = np.einsum('ij,ik,jk->k', coeffs, sinx, siny)
            return amplitude * out
        return f

    # Random polynomial generator
    def _polynomial():
        coeffs = np.random.randn(degree+1, degree+1) * amplitude
        def f(x):
            xx, yy = x[0], x[1]
            out = np.zeros_like(xx)
            for i in range(degree+1):
                for j in range(degree+1):
                    out += coeffs[i,j] * (xx**i) * (yy**j)
            return out
        return f

    # Gaussian radial basis function generator
    def _grbf():
        centers = np.random.rand(degree, 2)
        amps = np.random.randn(degree) * amplitude
        def f(x):
            xx, yy = x[0], x[1]
            out = np.zeros_like(xx)
            for (cx,cy), amp in zip(centers, amps):
                out += amp * np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*smoothness**2))
            return out
        return f

    # White noise generator
    def _white_noise():
        def f(x):
            return amplitude * np.random.randn(*x[0].shape)
        return f

    # Perlin noise generator
    def _perlin():
        try:
            from noise import pnoise2
        except ImportError:
            raise ImportError("Perlin noise requires the 'noise' package")
        def f(x):
            xx, yy = x[0], x[1]
            vec = np.vectorize(lambda xi, yi: pnoise2(xi/smoothness, yi/smoothness))
            return amplitude * vec(xx, yy)
        return f

    # Matérn GRF
    def _matern():
        # cov = gs.Matern(
        #     dim=2,
        #     var=var,
        #     len_scale=len_scale,
        #     nu=smoothness,
        # )
        # srf = gs.SRF(cov, mode="fft")
        # nx = ny = res + 1
        # grid = np.linspace(0.0, 1.0, nx)
        # field = srf.structured([grid, grid])
        # def f(x):
        #     xi = np.clip((x[0]*(nx-1)).astype(int), 0, nx-1)
        #     yi = np.clip((x[1]*(ny-1)).astype(int), 0, ny-1)
        #     return field[yi, xi]

        cov = gs.Matern(dim=2, var=var, len_scale=len_scale, nu=smoothness)
        srf = gs.SRF(cov, mode="fft")
        nx = ny = res + 1
        grid = np.linspace(0.0, 1.0, nx)
        field = srf.structured([grid, grid]).astype(np.float64)

        field -= field.mean()

        # Bilinear Interpolation
        def _bilinear_eval(F, xg, yg):
            x = np.clip(xg, 0.0, nx - 1.0)
            y = np.clip(yg, 0.0, ny - 1.0)
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            x1 = np.minimum(x0 + 1, nx - 1)
            y1 = np.minimum(y0 + 1, ny - 1)
            tx = x - x0
            ty = y - y0

            f00 = F[y0, x0]
            f10 = F[y0, x1]
            f01 = F[y1, x0]
            f11 = F[y1, x1]
            return ((1 - tx) * (1 - ty) * f00 +
                    tx * (1 - ty) * f10 +
                    (1 - tx) * ty * f01 +
                    tx * ty * f11)

        def f(x):
            xi = x[0] * (nx - 1)
            yi = x[1] * (ny - 1)
            return _bilinear_eval(field, xi, yi)

        return f


    # Exponential GRF
    def _exp():
        cov = gs.Exponential(
            dim=2,
            var=var,
            len_scale=len_scale,
        )
        srf = gs.SRF(cov, mode="fft")
        nx = ny = res + 1
        grid = np.linspace(0.0, 1.0, nx)
        field = srf.structured([grid, grid])
        def f(x):
            xi = np.clip((x[0]*(nx-1)).astype(int), 0, nx-1)
            yi = np.clip((x[1]*(ny-1)).astype(int), 0, ny-1)
            return field[yi, xi]
        return f

    # GRF
    def _grf():
        cov = gs.Gaussian(
            dim=2,
            var=var,
            len_scale=len_scale,
        )
        srf = gs.SRF(cov, mode="fft")
        nx = ny = res + 1
        grid = np.linspace(0.0, 1.0, nx)
        field = srf.structured([grid, grid])
        def f(x):
            xi = np.clip((x[0] * (nx - 1)).astype(int), 0, nx - 1)
            yi = np.clip((x[1] * (ny - 1)).astype(int), 0, ny - 1)
            return field[yi, xi]
        return f


    samplers = {
        'fourier': _fourier,
        'polynomial': _polynomial,
        'grbf': _grbf,
        'white_noise': _white_noise,
        'perlin': _perlin,
        'grf': _grf,
        'matern': _matern,
        'exp': _exp,
    }

    if type.lower() not in samplers:
        raise ValueError(f"Unknown sampler type '{type}'")

    return samplers[type.lower()]()


def gather_function_data(func, V, comm):
    """
    Gather a distributed dolfinx Function onto rank 0 and return a row‑major flat vector representation of func and
    a (ny, nx) grid representation of func.

    On non‑root ranks the function returns (None, None).

    Parameters
    ----------
    func : dolfinx.fem.Function
        The finite‑element function whose nodal values are to be extracted.
    V    : dolfinx.fem.FunctionSpace
        The function space associated with func.
    comm : mpi4py.MPI.Comm
        The communicator over which to gather.

    Returns
    -------
    f_vec  : np.ndarray or None
        Full global vector (row‑major) on rank 0, else None.
    f_grid : np.ndarray or None
        Full global grid on rank 0, else None.
    """
    # Local coordinates and values
    coords_local = V.tabulate_dof_coordinates()
    x_local = np.round(coords_local[:, 0], 12)
    y_local = np.round(coords_local[:, 1], 12)
    func.x.scatter_forward()
    vals_local = func.x.array.copy()
    local_data = np.column_stack((x_local, y_local, vals_local))

    # Gather to root
    gathered = comm.gather(local_data, root=0)

    if comm.rank == 0:
        data = np.vstack(gathered)

        # Build (x, y) key tuples, rounding to avoid floating–point mismatch
        pts = np.round(data[:, :2], 12)
        vals = data[:, 2]

        # Aggregate duplicates (shared DOFs across ranks) by averaging
        bucket = defaultdict(list)
        for (x_i, y_i), v in zip(map(tuple, pts), vals):
            bucket[(x_i, y_i)].append(v)

        coords_unique = np.array(list(bucket.keys()))
        vals_unique = np.array([np.mean(v_list) for v_list in bucket.values()])

        # Sort row‑major: y ascending, then x ascending
        sort_idx = np.lexsort((coords_unique[:, 0], coords_unique[:, 1]))
        coords_sorted = coords_unique[sort_idx]
        f_vec = vals_unique[sort_idx]

        unique_x = np.unique(coords_sorted[:, 0])
        unique_y = np.unique(coords_sorted[:, 1])
        nx, ny = unique_x.size, unique_y.size

        if nx * ny != len(f_vec):
            raise ValueError(
                f"After de‑duplicating DOFs we still cannot map onto a regular "
                f"{ny}×{nx} grid (have {len(f_vec)} values)."
            )

        f_grid = f_vec.reshape(ny, nx)

        return f_vec, f_grid

    # Non‑root ranks
    return None, None


def dict_collate(batch):
    """
    Collate a list of {"x":tensor, "y":tensor} into
    {"x":batch_tensor, "y":batch_tensor} using default_collate.
    """
    return {
        "x": default_collate([item["x"] for item in batch]),
        "y": default_collate([item["y"] for item in batch]),
    }


# %% 4. Discontinuous Galerkin -------------------------------------------------------------------------------
def legendre(x, k):
    """Vectorized Legendre polynomials"""
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    elif k == 2:
        return 0.5 * (3 * x ** 2 - 1)
    elif k == 3:
        return 0.5 * (5 * x ** 3 - 3 * x)
    else:
        raise ValueError("Polynomial order not implemented")

def dlegendre(x, k):
    """Vectorized Legendre derivatives"""
    if k == 0:
        return np.zeros_like(x)
    elif k == 1:
        return np.ones_like(x)
    elif k == 2:
        return 3 * x
    elif k == 3:
        return 7.5 * x ** 2 - 1.5
    else:
        raise ValueError("Polynomial order not implemented")

def limiter(U, dx, M=50.0, sensor_thresh=-3.0):
    """
    Fully vectorized TVB + Persson limiter.
    U: (N, p+1) array of DG modal coefficients per element.
    """
    U = U.copy()
    N, P1 = U.shape
    p = P1 - 1
    if p < 1:
        return U

    # TVB on linear mode (k=1)
    U0 = U[:, 0]
    a = U[:, 1]
    aL = 0.5 * (U0 - np.roll(U0, 1))
    aR = 0.5 * (np.roll(U0, -1) - U0)

    # Vectorized minmod
    same_sign = (np.sign(a) == np.sign(aL)) & (np.sign(a) == np.sign(aR))
    mm = np.where(
        same_sign,
        np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(aL), np.abs(aR))),
        0.0
    )

    keep_small = np.abs(a) <= M * dx * dx
    U[:, 1] = np.where(keep_small, a, mm)

    # Persson sensor on highest mode
    if p >= 2:
        num = U[:, p] ** 2
        den = np.sum(U * U, axis=1) + 1e-16
        S = np.log10(num / den)
        troubled = S > sensor_thresh
        U[troubled, p] = 0.0

    return U


def rhs(U, flux, PHI, WDPHI, PHI_L, PHI_R, M_inv):
    """
    Fully vectorized RHS computation for DG.
    U: (N, p+1) modal coefficients
    """
    # Volume term: reconstruct u at quad points for all elements
    u_vals = U @ PHI.T  # (N, quad_order)
    f_vals = flux(u_vals)

    # Volume contribution: -∫ f(u) * dphi dx
    RHS = -(f_vals @ WDPHI)  # (N, p+1)

    # Face traces (all elements at once)
    uL = U @ PHI_L  # Left face values
    uR = U @ PHI_R  # Right face values

    # Periodic neighbors
    uL_nb = np.roll(uR, 1)  # Left neighbor's right face
    uR_nb = np.roll(uL, -1)  # Right neighbor's left face

    # Vectorized Lax-Friedrichs flux
    alpha_L = np.maximum(np.abs(uL_nb), np.abs(uL))
    alpha_R = np.maximum(np.abs(uR), np.abs(uR_nb))

    F_L = 0.5 * (flux(uL_nb) + flux(uL) - alpha_L * (uL - uL_nb))
    F_R = 0.5 * (flux(uR) + flux(uR_nb) - alpha_R * (uR_nb - uR))

    # Surface contributions (broadcast efficiently)
    RHS += F_L[:, None] * PHI_L[None, :]
    RHS -= F_R[:, None] * PHI_R[None, :]

    # Mass matrix inverse (broadcast)
    RHS *= M_inv[None, :]

    return RHS

def compute_dt(U, dx, cfl, PHI, PHI_L, PHI_R):
    """Compute adaptive timestep based on CFL condition"""
    # Reconstruct solution at all quadrature points
    u_vals = U @ PHI.T

    # Also check face values
    uL = U @ PHI_L
    uR = U @ PHI_R

    # Maximum wave speed (Burgers: wave speed = |u|)
    u_max = max(np.max(np.abs(u_vals)),
                np.max(np.abs(uL)),
                np.max(np.abs(uR)))

    dt = cfl * dx / ((2 * U.shape[1]) * (u_max + 1e-10))
    dt = min(dt, 0.4 * dx)  # Safety cap

    return dt

def reconstruct_solution(u_coeffs, x_plot, x_edges, dx, legendre, d_res):
    """Efficiently reconstruct solution at plot points"""
    u_plot = np.zeros_like(x_plot)
    p = u_coeffs.shape[1] - 1

    # Find element for each plot point
    element_idx = np.floor((x_plot - x_edges[0]) / dx).astype(int)
    element_idx = np.clip(element_idx, 0, d_res - 1)

    # Process all elements at once using vectorization where possible
    for e in range(d_res):
        mask = element_idx == e
        if not np.any(mask):
            continue

        # Map to reference element [-1, 1]
        x_local = 2 * (x_plot[mask] - x_edges[e]) / dx - 1

        # Evaluate all basis functions at once
        phi_vals = np.zeros((len(x_local), p + 1))
        for k in range(p + 1):
            phi_vals[:, k] = legendre(x_local, k)

        # Reconstruct solution
        u_plot[mask] = phi_vals @ u_coeffs[e, :]

    return u_plot
