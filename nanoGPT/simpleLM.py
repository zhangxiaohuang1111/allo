import numpy as np
import allo
from allo.ir.types import float32
from allo.library.systolic import systolic
from allo.library.nn import (
    scaled_dot_product_attention,
    layer_norm,
    GeLU,
    residual_add,
)

def SimpleLanguageModel[
    Ty, L, D, Dffn, M0, M1
](
    X: "Ty[L, D]",
    Wq: "Ty[D, D]",
    Wk: "Ty[D, D]",
    Wv: "Ty[D, D]",
    Wp: "Ty[D, D]",
    W1: "Ty[D, Dffn]",
    W2: "Ty[Dffn, D]",
    gamma1: "Ty[D]",
    beta1: "Ty[D]",
    gamma2: "Ty[D]",
    beta2: "Ty[D]",
) -> "Ty[L, D]":
    
    # 自注意力机制
    Q: Ty[L, D] = 0
    K: Ty[L, D] = 0
    V: Ty[L, D] = 0

    # 使用 `systolic` 计算 Q, K, V
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "Q"](X, Wq, Q)
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "K"](X, Wk, K)
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "V"](X, Wv, V)

    # 计算 Self-Attention
    attn = scaled_dot_product_attention[Ty, 1, L, D, M0, M1](Q, K, V)

    # 输出密集层
    O_proj: Ty[L, D] = 0
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "P"](attn, Wp, O_proj)

    # 残差连接和 LayerNorm
    res_attn = residual_add[Ty, L, D, "res_attn"](O_proj, X)
    ln1 = layer_norm[Ty, L, D, "ln1"](res_attn, gamma1, beta1)

    # 前馈层
    ffn1: Ty[L, Dffn] = 0
    systolic[Ty, Ty, Ty, L, D, Dffn, M0, M1, "ffn1"](ln1, W1, ffn1)
    gelu_outp = GeLU[Ty, L, Dffn](ffn1)
    ffn2: Ty[L, D] = 0
    systolic[Ty, Ty, Ty, L, Dffn, D, M0, M1, "ffn2"](gelu_outp, W2, ffn2)

    # 最后一步残差连接和 LayerNorm
    res_ffn = residual_add[Ty, L, D, "res_ffn"](ffn2, ln1)
    output = layer_norm[Ty, L, D, "ln2"](res_ffn, gamma2, beta2)

    return output

def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def np_layernorm(inp, gamma, beta):
    mean = inp.mean(axis=1)
    mean2 = (inp**2).mean(axis=1)
    var = mean2 - mean**2
    np_out = gamma * (inp - mean[:, None]) / np.sqrt(var[:, None] + 1e-5) + beta
    return np_out

def np_simple_language_model(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2):
    # 1. 自注意力机制
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)
    
    # 1.1 Self-Attention
    attn_scores = np.matmul(Q, K.T)
    attn_probs = np_softmax(attn_scores)
    attn = np.matmul(attn_probs, V)
    
    # 1.2 输出密集层
    O_proj = np.matmul(attn, Wp)
    
    # 1.3 残差连接和 LayerNorm
    res_attn = O_proj + X
    ln1 = np_layernorm(res_attn, gamma1, beta1)
    
    # 2. 前馈层
    ffn1 = np.matmul(ln1, W1)
    gelu_outp = 0.5 * ffn1 * (1 + np.tanh(0.797885 * (ffn1 + 0.044715 * ffn1**3)))
    ffn2 = np.matmul(gelu_outp, W2)
    
    # 2.1 最后一步残差连接和 LayerNorm
    res_ffn = ffn2 + ln1
    output = np_layernorm(res_ffn, gamma2, beta2)
    
    return output

# 定义模型参数
L, D, Dffn = 8, 8, 16  # 确保与函数定义匹配
M0, M1 = 2, 2

# 调用 allo.customize 传入 float32 类型
s = allo.customize(
    SimpleLanguageModel,
    instantiate=[float32, L, D, Dffn, M0, M1],
)

mod = s.build(target="vitis_hls", mode="csim", project="simplelanguagemodel.prj")

# 测试输入和权重
X = np.random.randn(L, D).astype(np.float32)
Wq = np.random.randn(D, D).astype(np.float32)
Wk = np.random.randn(D, D).astype(np.float32)
Wv = np.random.randn(D, D).astype(np.float32)
Wp = np.random.randn(D, D).astype(np.float32)
W1 = np.random.randn(D, Dffn).astype(np.float32)
W2 = np.random.randn(Dffn, D).astype(np.float32)
gamma1 = np.random.randn(D).astype(np.float32)
beta1 = np.random.randn(D).astype(np.float32)
gamma2 = np.random.randn(D).astype(np.float32)
beta2 = np.random.randn(D).astype(np.float32)

# 使用 Allo 模型生成输出
allo_out = mod(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)

# 使用 NumPy 模型生成参考输出
np_out = np_simple_language_model(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)

# 断言输出一致
np.testing.assert_allclose(allo_out, np_out, atol=1e-2, rtol=1e-2)
print("Passed!")
# print(s.build(target="vhls"))