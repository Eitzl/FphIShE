import os

# Run Numba CUDA code in the simulator unless told otherwise. Must be set before numba is
# imported. On a real GPU, run `NUMBA_ENABLE_CUDASIM=0 pytest` to test the actual kernel.
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
