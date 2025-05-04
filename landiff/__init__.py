"""
LanDiff: A novel text-to-video generation framework that synergizes
Language Models and Diffusion Models.
"""

import os

from .utils import initialize_landiff_model_path

# Define version and other package metadata
__version__ = "0.1.0"

# Check if we should skip model initialization
skip_init = os.environ.get("LANDIFF_SKIP_INIT", "").lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

# Check if we should skip hash verification
skip_hash = os.environ.get("LANDIFF_SKIP_HASH_CHECK", "").lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

if not skip_init:
    # Pre-initialize the model path when the module is imported
    # This ensures the model is ready before any other functions are called
    try:
        # Initialize and verify the model path at import time
        model_path = initialize_landiff_model_path(skip_hash_verification=skip_hash)
        print(f"LanDiff model initialized at: {model_path}")
        if skip_hash:
            print("Hash verification was skipped due to LANDIFF_SKIP_HASH_CHECK environment variable.")
    except Exception as e:
        # Log any errors that occur during initialization
        # But don't raise, to allow other parts of the module to be used
        print(f"Warning: Failed to initialize LanDiff model path: {e}")
        print(
            "Some LanDiff functionality may not work until the model is properly set up."
        )
else:
    print(
        "LanDiff model initialization skipped due to LANDIFF_SKIP_INIT environment variable."
    )
