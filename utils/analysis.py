# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                              Utilities - Analysis Tools                                              #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from tabulate import tabulate

# %% 2. Functions --------------------------------------------------------------------------------------------
def _use_cmu_tex():
    """
    Configure Matplotlib to use LaTeX with CMU (Computer Modern Unicode) fonts.
    Requires a LaTeX distribution with the 'cmun' package installed (e.g., tlmgr install cmun).
    """
    import matplotlib as mpl
    import subprocess
    import shutil

    def have_tex() -> bool:
        return shutil.which("latex") is not None

    def have_pkg(sty: str) -> bool:
        kp = shutil.which("kpsewhich")
        if kp is None:
            return False
        try:
            # returncode == 0 means the package was found
            return subprocess.run([kp, sty], capture_output=True, text=True, check=False).returncode == 0
        except Exception:
            return False

    if have_tex() and have_pkg("cmun.sty"):
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["CMU Serif"],
            "text.latex.preamble": r"\usepackage{cmun}\usepackage[T1]{fontenc}\usepackage{amsmath}",
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        })
        print('CMUN.STY')
    elif have_tex() and have_pkg("lmodern.sty"):
        # Fallback to Latin Modern if CMU is not installed
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman"],
            "text.latex.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}",
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        })
        print('LMODERN.STY')
    else:
        # No LaTeX available: use mathtext with Computer Modern look
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        })
        print('mathtext')




def tensor_to_numpy(field):
    """
    Convert a PyTorch tensor or NumPy array to a 2D NumPy array by removing any leading channel dimensions.

    Parameters
    ----------
    field : torch.Tensor or numpy.ndarray
        Input array of shape (H, W) or with a leading channel dimension (C, H, W).

    Returns
    -------
    numpy.ndarray
        2D array of shape (H, W) with any singleton channel dimension removed.
    """
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    # Strip Leading Channel Dimension
    return field.squeeze()



def visualise_in_out(loader, data_processor, model, device, title, filename, n_samples, dpi=300):
    """
    Plot input, ground truth, model predictions, and errors (prediction − ground truth) for random samples.
    For 1D domains, shows line plots. For 2D domains, shows image plots with a per-axes colorbar on the right.
    Consistent color scaling is enforced across rows per column in the 2D case.
    """
    # Optional font setup (CMU via LaTeX with fallbacks)
    try:
        _use_cmu_tex()  # assumes you've defined the robust helper earlier
    except NameError:
        pass

    test_samples = loader.dataset
    if len(test_samples) == 0:
        print("Warning: Empty dataset - skipping visualization")
        return

    # Determine spatial dimension from first sample
    data0 = data_processor.preprocess(test_samples[0], batched=False)
    y0 = tensor_to_numpy(data0['y'])
    spatial_dim = 1 if y0.ndim == 1 else 2  # GT dimensionality

    indices = random.sample(range(len(test_samples)), min(n_samples, len(test_samples)))

    # -------------------------
    # 2D branch
    # -------------------------
    if spatial_dim == 2:
        # Determine number of input channels from a sample
        x0 = tensor_to_numpy(data_processor.preprocess(test_samples[0], batched=False)['x'])
        n_channels = x0.shape[0] if x0.ndim == 3 else 1
        n_cols = n_channels + 3  # inputs + GT + Pred + Err

        # Precompute values & global scales per column across all selected rows
        rows_cache = []  # list of tuples (x, y, out, err)
        min_in = [np.inf] * n_channels
        max_in = [-np.inf] * n_channels
        min_y = np.inf
        max_y = -np.inf
        min_out = np.inf
        max_out = -np.inf
        max_abs_err = 0.0

        for idx in indices:
            data = data_processor.preprocess(test_samples[idx], batched=False)
            x_t, y_t = data['x'], data['y']
            out_t = model(x_t.unsqueeze(0).to(device))[0]

            x = tensor_to_numpy(x_t)       # (C,H,W) or (H,W)
            y = tensor_to_numpy(y_t)       # (H,W)
            out = tensor_to_numpy(out_t)   # (H,W)
            err = out - y

            rows_cache.append((x, y, out, err))

            # update input scales
            if x.ndim == 3:
                for ch in range(n_channels):
                    vmin, vmax = float(np.min(x[ch])), float(np.max(x[ch]))
                    min_in[ch] = min(min_in[ch], vmin)
                    max_in[ch] = max(max_in[ch], vmax)
            else:
                vmin, vmax = float(np.min(x)), float(np.max(x))
                min_in[0] = min(min_in[0], vmin)
                max_in[0] = max(max_in[0], vmax)

            # gt / pred scales
            min_y = min(min_y, float(np.min(y)))
            max_y = max(max_y, float(np.max(y)))
            min_out = min(min_out, float(np.min(out)))
            max_out = max(max_out, float(np.max(out)))

            # symmetric error scale
            max_abs_err = max(max_abs_err, float(np.max(np.abs(err))))

        # Plot with consistent scales; put a colorbar on the RIGHT of each axes
        fig = plt.figure(constrained_layout=True, figsize=(4 * n_cols, 4 * len(indices)))

        for row_i, (x, y, out, err) in enumerate(rows_cache):
            # Inputs
            if x.ndim == 3:  # multi-channel
                for ch in range(n_channels):
                    ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + ch + 1)
                    im = ax.imshow(x[ch], vmin=min_in[ch], vmax=max_in[ch])
                    if row_i == 0:
                        ax.set_title(f'Input Ch{ch}', fontsize=16)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # per-axes colorbar on the right
                    cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.03, pad=0.02)
                    cbar.ax.tick_params(labelsize=12)
            else:  # single-channel
                ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + 1)
                im = ax.imshow(x, vmin=min_in[0], vmax=max_in[0])
                if row_i == 0:
                    ax.set_title('Input', fontsize=16)
                ax.set_xticks([])
                ax.set_yticks([])
                cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=12)

            # Ground Truth
            ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + n_channels + 1)
            im = ax.imshow(y, vmin=min_y, vmax=max_y)
            if row_i == 0:
                ax.set_title('Ground Truth', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=12)

            # Prediction
            ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + n_channels + 2)
            im = ax.imshow(out, vmin=min_out, vmax=max_out)
            if row_i == 0:
                ax.set_title('Prediction', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=12)

            # Error (symmetric, diverging)
            ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + n_channels + 3)
            im = ax.imshow(err, cmap='RdBu_r', vmin=-max_abs_err, vmax=max_abs_err)
            if row_i == 0:
                ax.set_title('Error', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=12)

        plt.savefig(filename, dpi=dpi)
        return

    # -------------------------
    # 1D branch (line plots)
    # -------------------------
    n_cols = 4
    fig = plt.figure(constrained_layout=True, figsize=(4 * n_cols, 4 * len(indices)))

    for row_i, idx in enumerate(indices):
        data = data_processor.preprocess(test_samples[idx], batched=False)
        x = data['x']; y = data['y']
        out = model(x.unsqueeze(0).to(device))[0]

        x = tensor_to_numpy(x)    # (C, L) or (L,)
        y = tensor_to_numpy(y)    # (L,)
        out = tensor_to_numpy(out)
        err = out - y

        length = y.shape[0]
        x_axis = np.linspace(0, 1, length)

        # Input
        ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + 1)
        if x.ndim == 2:  # multi-channel
            for ch in range(x.shape[0]):
                ax.plot(x_axis, x[ch], label=f'ch{ch}')
            if row_i == 0: ax.set_title('Input')
            if x.shape[0] > 1 and row_i == 0: ax.legend()
        else:
            ax.plot(x_axis, x)
            if row_i == 0: ax.set_title('Input')

        # Ground Truth
        ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + 2)
        ax.plot(x_axis, y)
        if row_i == 0: ax.set_title('Ground Truth')

        # Prediction
        ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + 3)
        ax.plot(x_axis, out)
        if row_i == 0: ax.set_title('Prediction')

        # Error
        ax = fig.add_subplot(len(indices), n_cols, row_i * n_cols + 4)
        ax.plot(x_axis, err)
        ax.axhline(0.0, linestyle='--', linewidth=0.8)
        if row_i == 0: ax.set_title('Error')

    plt.savefig(filename, dpi=dpi)


def surface_3d(field2d, *, x=None, y=None, title="3-D surface"):
    """
    Render and save a 3D surface plot of a 2D field using Plotly.
    """
    import plotly.graph_objects as go

    nz, nx = field2d.shape
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(nz)

    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=go.Surface(
        z=field2d,
        x=X,
        y=Y,
        colorscale="Viridis",
        showscale=True
    ))

    font_stack = "CMU Serif, Latin Modern Roman, STIXGeneral, Times New Roman, serif"
    fig.update_layout(
        title=dict(text=title, font=dict(family=font_stack, size=20)),
        font=dict(family=font_stack, size=14),  # default font (ticks, legend, etc.)
        scene=dict(
            xaxis=dict(
                title=dict(text="x", font=dict(family=font_stack, size=16)),
                tickfont=dict(family=font_stack, size=12)
            ),
            yaxis=dict(
                title=dict(text="y", font=dict(family=font_stack, size=16)),
                tickfont=dict(family=font_stack, size=12)
            ),
            zaxis=dict(
                title=dict(text="value", font=dict(family=font_stack, size=16)),
                tickfont=dict(family=font_stack, size=12)
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()
    fig.write_html(f"{title.replace(' ', '_')}.html")



def visualise_ind(loader, sample_idx=0, which="input", model=None, device="cpu"):
    """
    Visualize a single sample from the dataset in 3D for input, ground truth, or prediction.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader providing the dataset.
    sample_idx : int, optional
        Index of the sample to visualize (default 0).
    which : {'input', 'ground_truth', 'prediction'}, optional
        Type of surface to plot: 'input', 'ground_truth', or 'prediction' (default 'input').
    model : torch.nn.Module, optional
        Trained model used to generate predictions (required if which is 'prediction').
    device : str or torch.device, optional
        Device on which to run the model (default 'cpu').

    Returns
    -------
    None
        Displays the 3D surface in a browser and writes an HTML file.
    Raises
    ------
    ValueError
        If which is not one of 'input', 'ground_truth', or 'prediction'.
    """
    sample = loader.dataset[sample_idx]

    if isinstance(sample, dict):                 # neuralop style
        x_tensor, y_tensor = sample["x"], sample["y"]
    else:                                        # ordinary tuple dataset
        x_tensor, y_tensor = sample

    # to (H,W) numpy
    x_img = tensor_to_numpy(x_tensor[0])
    y_img = tensor_to_numpy(y_tensor[0])

    if which == "input":
        #surface_3d(x_img, title=f"3D Input {sample_idx}")
        x_tensor = sample["x"]
        if x_tensor.ndim == 3 and x_tensor.shape[0] > 1:
            for ch in range(x_tensor.shape[0]):
                surface_3d(tensor_to_numpy(x_tensor[ch]), title=f"3D Input Channel {ch} {sample_idx}")
        else:
            surface_3d(x_img, title=f"3D Input {sample_idx}")

    elif which == "ground_truth":
        surface_3d(y_img, title=f"3D Ground Truth {sample_idx}")

    elif which == "prediction":
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor.unsqueeze(0).to(device))[0]
        surface_3d(tensor_to_numpy(pred), title=f"3D Prediction {sample_idx}")

    else:
        raise ValueError("which must be 'input', 'ground_truth', or 'prediction'.")




def visualise_1d_sample(test_loader, data_processor, model, device, sample_idx, title, filename, dpi=300):
    """
    Plot a single 1D sample: input, ground truth, prediction.
    """
    import torch
    model.eval()

    # Fetch raw sample and preprocess
    raw_sample = test_loader.dataset[sample_idx]
    if hasattr(data_processor, "preprocess"):
        data = data_processor.preprocess(raw_sample, batched=False)
        x_single = data["x"]
        y_single = data["y"]
    else:
        if isinstance(raw_sample, dict):
            x_single, y_single = raw_sample["x"], raw_sample["y"]
        else:
            x_single, y_single = raw_sample

    # Forward pass
    with torch.no_grad():
        pred_single = model(x_single.unsqueeze(0).to(device))[0].detach().cpu()

    # To numpy
    x_np = tensor_to_numpy(x_single)
    y_np = tensor_to_numpy(y_single)
    pred_np = tensor_to_numpy(pred_single)

    # Build x-axis over [0,1]
    length = y_np.shape[-1]
    x_axis = np.linspace(0, 1, length)

    # Make figure: 3 columns — Input | Ground Truth | Prediction
    n_cols = 3
    fig = plt.figure(figsize=(4 * n_cols, 3.2))

    # Input (handle multi-channel if present)
    ax = fig.add_subplot(1, n_cols, 1)
    if x_np.ndim == 2:  # (C, L)
        for ch in range(x_np.shape[0]):
            ax.plot(x_axis, x_np[ch], label=f"ch{ch}")
        ax.legend(loc="best")
    else:               # (L,)
        ax.plot(x_axis, x_np)
    ax.set_title("Input")
    ax.set_xlabel("x")
    ax.set_ylabel("value")

    # Ground Truth
    ax = fig.add_subplot(1, n_cols, 2)
    ax.plot(x_axis, y_np)
    ax.set_title("Ground Truth")
    ax.set_xlabel("x")

    # Prediction
    ax = fig.add_subplot(1, n_cols, 3)
    ax.plot(x_axis, pred_np)
    ax.set_title("Prediction")
    ax.set_xlabel("x")

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)

def evaluate_model_2d(model, test_loader, title, filename, device, visualise, data_processor,
                      eval_size=None, resize_mode='bilinear', align_corners=False):
    """
    Compute diagnostic metrics for an FNO model and print them in a formatted table.

    Parameters
    ----------
    model : torch.nn.Module
        Trained FNO model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    device : str or torch.device, optional
        Device to perform computation on (default 'cpu').
    visualise : bool, optional
        If True, generate 3D visualizations of a random test sample (default False).
    data_processor : object, optional
        DataProcessor with a preprocess method, required for visualisation.

    Returns
    -------
    None
        Prints test L2, Linf, H1 errors, model parameter count, and average inference time.
    """
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tabulate import tabulate

    model = model.to(device)
    model.eval()

    # Loss Functions
    l2_fn = LpLoss(d=2, p=2)
    linf_fn = LpLoss(d=2, p=float('inf'))
    h1_fn = H1Loss(d=2)

    warned_target_resize = False

    def maybe_resize(t: torch.Tensor, size):
        """Resize BCHW tensor to size=(H,W) if needed."""
        if size is None or t.shape[-2:] == size:
            return t
        return F.interpolate(t, size=size, mode=resize_mode, align_corners=align_corners)

    def compute_loss(loader):
        nonlocal warned_target_resize

        rel_l2s, rel_linfs, rel_h1s = [], [], []
        model.eval()
        with torch.no_grad():
            for sample in loader:
                if isinstance(sample, dict):
                    x, y = sample["x"].to(device), sample["y"].to(device)
                else:
                    x, y = sample
                    x, y = x.to(device), y.to(device)

                # Resize input to eval grid
                x_in = maybe_resize(x, eval_size)

                pred = model(x_in)

                target_size = pred.shape[-2:]
                if y.shape[-2:] != target_size:
                    if not warned_target_resize:
                        print(f"[warn] Resizing targets from {tuple(y.shape[-2:])} "
                              f"to {target_size} to match eval_size and compute loss.")
                        warned_target_resize = True
                    y = maybe_resize(y, target_size)

                l2 = l2_fn(pred, y, relative=True).item()
                linf = linf_fn(pred, y, relative=True).item()
                h1 = h1_fn(pred, y, relative=True).item()

                rel_l2s.append(l2)
                rel_linfs.append(linf)
                rel_h1s.append(h1)

        return float(np.mean(rel_l2s)), float(np.mean(rel_linfs)), float(np.mean(rel_h1s))

    # Compute Losses
    test_l2, test_linf, test_h1 = compute_loss(test_loader)

    # Count Parameters
    num_params = count_model_params(model)

    # Inference Time per Batch
    n_batches = 0
    total_time = 0.0
    with torch.no_grad():
        for sample in test_loader:
            if isinstance(sample, dict):
                x = sample["x"].to(device)
            else:
                x, _ = sample
                x = x.to(device)

            x_in = maybe_resize(x, eval_size)

            start = time.time()
            _ = model(x_in)
            total_time += (time.time() - start)
            n_batches += 1
    avg_infer_ms = (total_time / n_batches) * 1000 if n_batches > 0 else float('nan')

    # Visualisation
    test_samples = test_loader.dataset
    sample_idx = random.choice(range(len(test_samples)))

    if visualise:
        visualise_ind(test_loader, sample_idx=sample_idx, which="input", model=model, device=device)
        visualise_ind(test_loader, sample_idx=sample_idx, which="ground_truth", model=model, device=device)
        visualise_ind(test_loader, sample_idx=sample_idx, which="prediction", model=model, device=device)
        visualise_in_out(test_loader, data_processor=data_processor, model=model, device=device,
                         title=title, filename=filename, n_samples=3)

    # Results Table
    table = [
        ["Metric",               "Value"],
        ["Test L2 Error",        f" {100 * test_l2 :.2f}%"],
        ["Test Linf Error",      f" {100 * test_linf :.2f}%"],
        ["Test H1 Error",        f" {100 * test_h1 :.2f}%"],
        ["# Parameters",         f"{num_params:,}"],
        ["Avg. inference (ms)",  f"{avg_infer_ms:.2f}"],
    ]

    print("\nModel diagnostics:\n")
    print(tabulate(table, headers="firstrow", tablefmt="github"))


def evaluate_model_1d(model, test_loader, title, filename, device, visualise, data_processor, n_vis_samples=3):
    """
    Compute diagnostic metrics for an FNO model and print them in a formatted table.

    Parameters
    ----------
    model : torch.nn.Module
        Trained FNO model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    device : str or torch.device, optional
        Device to perform computation on (default 'cpu').
    visualise : bool, optional
        If True, generate visualizations (for 1D: stacked line plots of random samples).
    data_processor : object, optional
        DataProcessor with a preprocess method, required for visualisation.
    n_vis_samples : int, optional — Number of random samples to visualize in a stacked (rows) layout (default 3).

    Returns
    -------
    None
        Prints test L2, Linf, H1 errors, model parameter count, and average inference time.
    """
    import time
    import numpy as np
    import torch

    from tabulate import tabulate

    model = model.to(device)
    model.eval()

    # Loss Functions
    l2_fn = LpLoss(d=2, p=2)
    linf_fn = LpLoss(d=2, p=float('inf'))
    h1_fn = H1Loss(d=2)

    def compute_loss(loader):
        rel_l2s, rel_linfs, rel_h1s = [], [], []
        model.eval()
        with torch.no_grad():
            for sample in loader:
                if isinstance(sample, dict):
                    x, y = sample["x"].to(device), sample["y"].to(device)
                else:
                    x, y = sample
                    x, y = x.to(device), y.to(device)

                pred = model(x)

                l2 = l2_fn(pred, y, relative=True).item()
                linf = linf_fn(pred, y, relative=True).item()
                h1 = h1_fn(pred, y, relative=True).item()

                rel_l2s.append(l2)
                rel_linfs.append(linf)
                rel_h1s.append(h1)

        return float(np.mean(rel_l2s)), float(np.mean(rel_linfs)), float(np.mean(rel_h1s))

    # Compute Losses
    test_l2, test_linf, test_h1 = compute_loss(test_loader)

    # Count Parameters
    num_params = count_model_params(model)

    # Inference Time per Batch
    n_batches = 0
    total_time = 0.0
    with torch.no_grad():
        for sample in test_loader:
            if isinstance(sample, dict):
                x = sample["x"].to(device)
            else:
                x, _ = sample
                x = x.to(device)

            start = time.time()
            _ = model(x)
            total_time += (time.time() - start)
            n_batches += 1
    avg_infer_ms = (total_time / n_batches) * 1000 if n_batches > 0 else float('nan')

    # Visualisation
    test_samples = test_loader.dataset
    sample_idx = random.choice(range(len(test_samples)))

    if visualise:
        visualise_in_out(test_loader, data_processor=data_processor, model=model, device=device,
                         title=title, filename=filename, n_samples=n_vis_samples)


    # Results Table
    table = [
        ["Metric",               "Value"],
        ["Test L2 Error",        f" {100 * test_l2 :.2f}%"],
        ["Test Linf Error",      f" {100 * test_linf :.2f}%"],
        ["Test H1 Error",        f" {100 * test_h1 :.2f}%"],
        ["# Parameters",         f"{num_params:,}"],
        ["Avg. inference (ms)",  f"{avg_infer_ms:.2f}"],
    ]

    print("\nModel diagnostics:\n")
    print(tabulate(table, headers="firstrow", tablefmt="github"))


def model_summary(verbose, title, model, hidden_channels, n_modes, n_layers, lr, weight_decay, betas, eps, step_size,
                  gamma, train_loss, l2_loss, linf_loss, h1_loss, res, N_samples, train_idx, valid_idx, test_idx,
                  device, epochs, batch_size, eval_interval, num_workers):
    """
    Print a detailed summary of the FNO model, its training configuration, and dataset statistics.

    Prints sections on:
    - Model architecture and parameter distribution
    - Optimizer and scheduler configuration
    - Loss functions used
    - Data resolution and sample splits
    - Hardware and training hyperparameters

    Parameters
    ----------
    verbose : bool
        If True, prints all summary information.
    title : str
        Heading printed before the summary.
    model : torch.nn.Module
        FNO model to summarize.
    hidden_channels : int
        Number of channels in the model's hidden layers.
    n_modes : int
        Number of Fourier modes per dimension.
    n_layers : int
        Number of Fourier layers in the model.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay factor.
    betas : tuple of float
        Beta parameters for the optimizer.
    eps : float
        Epsilon parameter for the optimizer.
    step_size : int
        Step size for the learning rate scheduler.
    gamma : float
        Decay factor for the learning rate scheduler.
    train_loss : callable
        Loss function used during training.
    l2_loss : callable
        L2 loss used for validation.
    linf_loss : callable
        Linf loss used for validation.
    h1_loss : callable
        H1 loss used for validation.
    res : int
        Spatial resolution (grid size).
    N_samples : int
        Total number of samples in the dataset.
    train_idx : sequence
        Indices for the training set.
    valid_idx : sequence
        Indices for the validation set.
    test_idx : sequence
        Indices for the test set.
    device : str or torch.device
        Device for computation.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    eval_interval : int
        Epoch interval for evaluation.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    None
        All information is printed to standard output.
    """

    if verbose:
        print(title)

        # Model summary: layer shapes and parameter counts
        print("\n## Model Architecture ##")
        print(f"🧬 Model type: FNO-2D")
        print(f"🎛️ Hidden channels: {hidden_channels}")
        print(f"🔉 Number of Fourier modes: {n_modes} (per dimension)")
        print(f"🍰 Number of layers: {n_layers}")

        print("\nParameter Distribution:")

        def prettify(name):
            return name.replace('_', ' ').replace('.', ' ').title()

        param_table = []
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                human_name = prettify(name)
                num_params = param.numel()
                total_params += num_params
                param_table.append([human_name, str(list(param.shape)), f"{num_params:,}"])

        param_table.append(["Total Trainable Parameters", "", f"{total_params:,}"])
        print(tabulate(param_table, headers=["Parameter", "Shape", "Params"], tablefmt="github"))

        print(f"🎛️ Model Parameters (Including Non-Trainable): {count_model_params(model):,}")

        # Optimizer configuration
        print("\n## Optimser Configuration ##")
        print(f"Optimizer: AdamW")
        print(f"Learning rate: {lr:.1e}")
        print(f"Weight decay: {weight_decay:.1e}")
        print(f"Betas: {betas}")
        print(f"Epsilon: {eps:.1e}")

        # Scheduler details
        print("\n## Learning Rate Scheduler ##")
        print(f"Type: StepLR")
        print(f"Step size: {step_size} epochs")
        print(f"Gamma (decay factor): {gamma}")

        # Loss functions
        print("\n## Loss Functions ##")
        loss_table = [
            ["Training", train_loss.__class__.__name__, "L2 (relative)"],
            ["Validation - L2", l2_loss.__class__.__name__, "L2 (relative)"],
            ["Validation - H1", h1_loss.__class__.__name__, "H1 (relative)"],
            ["Validation - Linf", linf_loss.__class__.__name__, "L∞ (relative)"]
        ]
        print(tabulate(loss_table, headers=["Phase", "Loss Type", "Metric"], tablefmt="github"))

        # Data statistics
        print("\n## Data Statistics ##")
        print(f"Domain Resolution: {res:,}")
        print(f"Total samples: {N_samples:,}")
        print(f"Training samples: {len(train_idx):,} ({len(train_idx) / N_samples:.1%})")
        print(f"Validation samples: {len(valid_idx):,} ({len(valid_idx) / N_samples:.1%})")
        print(f"Test samples: {len(test_idx):,} ({len(test_idx) / N_samples:.1%})")

        # Device and precision info
        print("\n## Hardware Configuration ##")
        print(f"Device: {device}")
        print(f"Torch version: {torch.__version__}")

        # Training configuration
        print("\n## Training Configuration ##")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Evaluation interval: every {eval_interval} epochs")
        print(f"Number of workers: {num_workers}")

        return