"""
constants.py

Central registry of numeric constants EXPLICITLY documented across the 19 context
files, so no single Model file needs to silently redefine them. Anything not listed
here and required by a Model/Controller must be supplied by the caller explicitly
(no invented defaults), per the anti-hallucination rules.
"""

# Preprocessing and Patching (SR).md
PATCH_SIZE = 128
PATCH_STRIDE = 96
PATCH_OVERLAP = 32

# Shallow Convolution.md / Hybrid Block.md
BASE_CHANNELS = 64

# Hybrid Block.md hyperparameters table
DEFAULT_NUM_HYBRID_BLOCKS = 8
GROWTH_RATE = 32
NUM_DENSE_LAYERS = 5
NUM_RDB_PER_RRDB = 3
RESIDUAL_BETA = 0.2
LEAKY_RELU_SLOPE = 0.2
SWIN_WINDOW_SIZE = 8
SWIN_NUM_HEADS = 4
SWIN_MLP_HIDDEN = 256

# Upsampling.md
PIXEL_SHUFFLE_R = 2
ANCHOR_IMAGE_SIZE = 320

# Latent.md / Noise Added.md
LATENT_CHANNELS = 4
LATENT_SPATIAL_SIZE = 40
DDPM_T = 1000
DDPM_BETA_MIN = 1e-4
DDPM_BETA_MAX = 2e-2
T_PRIME_MIN = 1
T_PRIME_MAX = 16

# Original-Image Cond. Features.md
COND_CHANNELS_DEFAULT = 256

# Shared across Latent.md / VAE Decoder.md / Original-Image Cond. Features.md
GROUP_NORM_GROUPS = 32

# PatchGAN Discriminator.md
PATCHGAN_RECEPTIVE_FIELD = 70

# Uncertainty Estimation.md (documented only as an example value, "e.g. N=10")
STRUCTURAL_UNCERTAINTY_N_EXAMPLE = 10