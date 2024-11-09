import allo
import numpy as np
import pytest
from allo.ir.types import int32, float32
import time
random_seed = 0
np.random.seed(random_seed)
# Define a 1D convolution kernel with input size 16 and filter size 7
def conv1D(A: float32[512], W: float32[7]) -> float32[506]:
    B: float32[506] = 0.0
    for x in allo.grid(506): 
        v: float32 = 0.0
        for r in allo.reduction(7):  #
            v += A[x + r] * W[r]  # 
        B[x] = v
    return B


# Customize and optimize the convolution function
s = allo.customize(conv1D)
s.reuse_at(s.A, "x")  # Reuse A in the x direction (line buffer optimization)
s.pipeline("x")  # Pipeline the outer loop for better performance
# print(s.module)
#s.to fifo

# Build the module for Vitis HLS C simulation (CSIM)
# mod = s.build(target="vitis_hls", mode="csim", project="conv_1D_big.prj")
# print("CSIM Code Generated")

# mod = s.build(target="llvm")
# print("llvm Code Generated")

# print(mod)
mod = s.build(target="vitis_hls", mode="sw_emu", project="conv_1D_big.prj")
print("sw_emu Code Generated")

# mod = s.build(target="vitis_hls", mode="hw_emu", project="conv_1D_big.prj")
# print("hw_emu Code Generated")

# mod = s.build(target="vitis_hls", mode="csyn", project="conv_1D_big.prj")
# print("csyn Code Generated")

# mod = s.build(target="vitis_hls", mode="hw", project="conv_1D_big.prj")
# print("bitstream Generated")
# Prepare input and output data
np_A = np.random.rand(512).astype(np.float32)  # 输入数据（长度 512）
np_W = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)  # 1D 卷积滤波器（长度 7）
np_B = np.zeros(506, dtype=np.float32)  # 输出缓冲区（长度 506）
#---------------------------------------------------------------
# num_runs = 10000  # You can adjust this number
# execution_times = []
# # Run the module multiple times
# for _ in range(num_runs):
#     start_time = time.time()
#     np_B = mod(np_A, np_W)  # Call the compiled module
#     end_time = time.time()
    
#     # Calculate elapsed time for each run and store it
#     elapsed_time = end_time - start_time
#     execution_times.append(elapsed_time)

# # Calculate the average execution time in seconds
# average_time_sec = sum(execution_times) / num_runs

# # Convert the average execution time to nanoseconds
# average_time_ns = average_time_sec * 1e9

# # Print the results
# print(f"Average execution time over {num_runs} runs: {average_time_ns:.2f} ns")
#---------------------------------------------------------------


mod(np_A, np_W, np_B)  
# Check if the results are close to the expected result
expected_C = np.correlate(np_A, np_W)
np.testing.assert_allclose(np_B, expected_C, rtol=1e-5, atol=1e-5)

print("Test Passed! The results match numpy correlate.")
