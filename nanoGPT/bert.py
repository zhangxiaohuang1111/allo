import numpy as np
import allo
from allo.ir.types import int8, float32
import pytest
from allo.library.systolic import systolic
from allo.library.nn import (
    scaled_dot_product_attention,
    layer_norm,
    GeLU,
    residual_add,
)

H, L, D, Dffn = 2, 8, 8, 16
M0, M1 = 2, 2

def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sdp(Q, K, V, H, D):
    context = np.zeros(Q.shape)
    h_d = D // H
    for i in range(H):
        # split Q, K, V
        Q_h = Q[:, i * h_d : (i + 1) * h_d]
        K_h = K[:, i * h_d : (i + 1) * h_d]
        V_h = V[:, i * h_d : (i + 1) * h_d]
        # compute attention
        attention = np.matmul(Q_h, K_h.T)
        Y = np_softmax(attention)
        context_i = np.matmul(Y, V_h)
        context[:, i * h_d : (i + 1) * h_d] = context_i
    return context

def np_layernorm(inp, gamma, beta):
    mean = inp.mean(axis=1)
    mean2 = (inp**2).mean(axis=1)
    var = mean2 - mean**2
    np_out = gamma * (inp - mean[:, None]) / np.sqrt(var[:, None] + 1e-5) + beta
    return np_out

def bert_layer(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2):
    # 1. Bert Attention
    # 1.0 project Q, K, V
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)
    # 1.1 self attention
    attn = sdp(Q, K, V, H, D)
    # 1.2 output dense
    O_proj = np.matmul(attn, Wp)
    # 1.3 Residual layer
    res_attn = O_proj + X
    # 1.4 layer norm
    ln = np_layernorm(res_attn, gamma1, beta1)
    # 2. Feed Forward Network
    # 2.1 ffn dense 1
    ffn1 = np.matmul(ln, W1)
    # 2.2 gelu layer
    gelu_outp = 0.5 * ffn1 * (1 + np.tanh(0.797885 * (ffn1 + 0.044715 * ffn1**3)))
    # 2.3 ffn dense 2
    ffn2 = np.matmul(gelu_outp, W2)
    # 2.4 Residual layer
    res_ffn = ffn2 + ln
    # 2.5 layer norm
    ffn_ln_outp = np_layernorm(res_ffn, gamma2, beta2)
    return ffn_ln_outp


def BertLayer[
    Ty, H, L, D, Dffn, M0, M1
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
    # 1. Bert Attention
    # 1.0 project Q, K, V
    Q: Ty[L, D] = 0
    K: Ty[L, D] = 0
    V: Ty[L, D] = 0
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "Q"](X, Wq, Q)
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "K"](X, Wk, K)
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "V"](X, Wv, V)
    # 1.1 self attention
    attn = scaled_dot_product_attention[Ty, H, L, D, M0, M1](Q, K, V)
    # 1.2 output dense
    O_proj: Ty[L, D] = 0
    systolic[Ty, Ty, Ty, L, D, D, M0, M1, "P"](attn, Wp, O_proj)
    # 1.3 Residual layer
    res_attn = residual_add[Ty, L, D, "res_attn"](O_proj, X)
    # 1.4 layer norm
    ln = layer_norm[Ty, L, D, "ln1"](res_attn, gamma1, beta1)
    # 2. Feed Forward Network
    # 2.1 ffn dense 1
    ffn1: Ty[L, Dffn] = 0
    systolic[Ty, Ty, Ty, L, D, Dffn, M0, M1, "ffn1"](ln, W1, ffn1)
    # 2.2 gelu layer
    gelu_outp = GeLU[Ty, L, Dffn](ffn1)
    # 2.3 ffn dense 2
    ffn2: Ty[L, D] = 0
    systolic[Ty, Ty, Ty, L, Dffn, D, M0, M1, "ffn2"](gelu_outp, W2, ffn2)
    # 2.4 Residual layer
    res_ffn = residual_add[Ty, L, D, "res_ffn"](ffn2, ln)
    # 2.5 layer norm
    ffn_ln_outp = layer_norm[Ty, L, D, "ln2"](res_ffn, gamma2, beta2)
    return ffn_ln_outp

s = allo.customize(
    BertLayer,
    instantiate=[float32, H, L, D, Dffn, M0, M1],
)
mod = s.build(target="vitis_hls", mode="csim", project="bert.prj")
X = np.random.randn(L, D).astype(np.float32)
# weights are supposed to be transposed
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
allo_out = mod(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)


np_out = bert_layer(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)
np.testing.assert_allclose(allo_out, np_out, atol=1e-2, rtol=1e-2)
print("Passed!")