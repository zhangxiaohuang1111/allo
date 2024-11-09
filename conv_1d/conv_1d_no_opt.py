import allo
import numpy as np
import pytest
from allo.ir.types import int32, float32

# Define a 1D convolution kernel
def conv1D(A: float32[16], W: float32[3]) -> float32[14]:
    B: float32[14] = 0.0
    for x in allo.grid(14):  # Output grid size is 14
        v: float32 = 0.0
        for r in allo.reduction(3):  # 3-element convolution kernel
            v += A[x + r] * W[r]  # Apply convolution
        B[x] = v
    return B

# Customize and optimize the convolution function
s = allo.customize(conv1D)
# s.reuse_at(s.A, "x")  # Reuse A in the x direction (line buffer optimization)
# s.pipeline("x")  # Pipeline the outer loop for better performance
# print(s.module)
#s.to fifo
# Build the module for Vitis HLS C simulation (CSIM)
# mod = s.build(target="vitis_hls", mode="csim", project="conv1D_no_opt.prj")
# print("CSIM Code Generated")

# mod = s.build(target="vitis_hls", mode="sw_emu", project="conv1D_no_opt.prj")
# print("sw_emu Code Generated")

# mod = s.build(target="vitis_hls", mode="hw_emu", project="conv1D_no_opt.prj")
# print("hw_emu Code Generated")

mod = s.build(target="vitis_hls", mode="csyn", project="conv1D_no_opt.prj")
print("csyn Code Generated")

# mod = s.build(target="vitis_hls", mode="hw", project="conv1D_no_opt.prj")
# print("bitstream Generated")
# Prepare input and output data
np_A = np.random.rand(16).astype(np.float32)  # Input data (length 16)
np_W = np.array([0.25, 0.5, 0.25], dtype=np.float32)  # 1D Convolution filter (length 3)
np_B = np.zeros(14, dtype=np.float32)  # Output buffer (length 14)

# Run the compiled module and get the output
mod(np_A, np_W, np_B)  # Pass np_B as the output buffer

# Now np_B contains the result of the convolution
# print("Output Y:", np_B)

# Check if the results are close to the expected result
expected_C = np.convolve(np_A, np_W, mode='valid')
np.testing.assert_allclose(np_B, expected_C, rtol=1e-5, atol=1e-5)

print("Test Passed! The results match NumPy's convolution.")
