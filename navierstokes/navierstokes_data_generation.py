# -------------------------------------------------------------------------------------------------------------------- #
#                                               Fourier Neural Operators                                               #
#                                     2D Navier Stokes Equation - Data Generation                                      #
# -------------------------------------------------------------------------------------------------------------------- #


# %% 1. Preamble ---------------------------------------------------------------------------------------------
# File Path
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dependencies
import numpy as np
import matplotlib.pyplot as plt


# Parameters




# %% 2. Pseudospectral Setup ---------------------------------------------------------------------------------
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)



