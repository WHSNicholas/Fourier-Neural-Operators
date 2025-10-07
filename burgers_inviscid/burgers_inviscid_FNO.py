# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                      1D Viscous Burgers Equation - FNO Training                                      #
# -------------------------------------------------------------------------------------------------------------------- #

# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Dependencies
import torch
from utils.data import dict_collate, Dataset
from utils.analysis import evaluate_model_1d, model_summary
from torch.utils.data import DataLoader, Subset
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.models import FNO
from neuralop.data.transforms import data_processors


# Parameters
# Basic
verbose = True                                    # Verbosity
eval = True                                       # Model Evaluation
epochs = 500                                      # Epochs
res = 2**9                                        # Downsampling Resolution
filename = f"burgers_inviscid_model_{res}_huge"   # Filename

batch_size = 4                                    # Batch Size
num_workers = 12                                  # Parallelising

# FNO Model
i, o = 1, 1                                       # Input/Output Dimension
hidden_channels = 48                              # Dimension of Latent Representation
n_modes = 16                                      # Number of Fourier Modes
n_layers = 6                                      # Number of Layers
d = 1                                             # Spatial Domain
p = 2                                             # Lp Loss

# Optimiser
lr = 10**-3                                       # Learning Rate
betas = (0.9, 0.999)                              # Decay Rates for Moments
eps = 10**-6                                      # Epsilon for Stability
weight_decay = 10**-4                             # Weight Decay
step_size = 100                                   # Learning Rate Step Decay
gamma = 0.5                                       # Learning Rate Decay

# Trainer
wandb_log = False                                 # Weights and Biases Log
eval_interval = 25                                # Evaluation Interval
use_distributed = False                           # Distributed Runtime
train = True                                      # Train Model

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %% 2. Preprocessing ----------------------------------------------------------------------------------------
# Instantiate HDF5-backed dataset
full_dataset = Dataset("burgers_inviscid/burgers_inviscid_dataset.h5", res=res)

# Train Val Test Split
N_samples = len(full_dataset)
perm = torch.randperm(N_samples)
N_train = int(5/6 * N_samples)
N_valid = N_test = int(1/12 * N_samples)

train_idx = perm[ : N_train]
valid_idx = perm[N_train : N_train + N_valid]
test_idx  = perm[N_train + N_valid : ]

train_data = Subset(full_dataset, train_idx)
valid_data = Subset(full_dataset, valid_idx)
test_data = Subset(full_dataset, test_idx)

# DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=dict_collate,
)

val_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=dict_collate,
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=dict_collate,
)


# %% 4. Model ------------------------------------------------------------------------------------------------
# Creating FNO Model
model = FNO(
    in_channels=i,
    out_channels=o,
    hidden_channels=hidden_channels,
    n_modes=(n_modes, ),
    n_layers=n_layers,
)

model = model.to(device)
data_processor = data_processors.DefaultDataProcessor().to(device)

# Optimiser
optimiser = AdamW(
    model.parameters(),
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimiser,
    step_size=step_size,
    gamma=gamma,
)

# Loss Functions
l2_loss = LpLoss(d, p)
h1_loss = H1Loss(d)
linf_loss = LpLoss(d, p=float('inf'))

train_loss = l2_loss
eval_loss={'H1': h1_loss, 'L2': l2_loss, 'Linf': linf_loss}

model_summary(verbose=verbose, title="#### FNO 1D-INVISCID BURGERS ####",
              model=model, hidden_channels=hidden_channels, n_modes=n_modes, n_layers=n_layers, lr=lr,
              weight_decay=weight_decay, betas=betas, eps=eps, step_size=step_size, gamma=gamma, train_loss=train_loss,
              l2_loss=l2_loss, linf_loss=linf_loss, h1_loss=h1_loss, res=res, N_samples=N_samples, train_idx=train_idx,
              valid_idx=valid_idx, test_idx=test_idx, device=device, epochs=epochs, batch_size=batch_size,
              eval_interval=eval_interval, num_workers=num_workers)

# %% 5. Training FNO -----------------------------------------------------------------------------------------
# Training
trainer = Trainer(
    model=model,
    n_epochs=epochs,
    device=device,
    data_processor=data_processor,
    wandb_log=wandb_log,
    eval_interval=eval_interval,
    use_distributed=use_distributed,
    verbose=verbose,
)

if train:
    trainer.train(
        train_loader=train_loader,
        test_loaders={"val": val_loader},
        optimizer=optimiser,
        scheduler=scheduler,
        training_loss=train_loss,
        eval_losses=eval_loss,
    )

    torch.save(model.state_dict(), f"burgers_inviscid/{filename}.pt")
    print(f"✅ Trained model saved to 'burgers_inviscid/{filename}.pt'")

if not train:
    model.load_state_dict(torch.load(f"burgers_inviscid/{filename}.pt", map_location=device, weights_only=False))


# %% 5. Model Evaluation ----------------------------------------------------------------------------------------
if eval:
    evaluate_model_1d(model, test_loader, device=device, visualise=True, data_processor=data_processor,
                      title='FNO for 1D Inviscid Burgers', filename='burgers_inviscid_io.png', n_vis_samples=3)