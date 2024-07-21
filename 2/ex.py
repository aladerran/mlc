import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T



# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)

# numpy version
c_np = a + b
c_np

@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4,4), "int64"),
          B: T.Buffer((4), "int64"),
          C: T.Buffer((4,4), "int64")):
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    for i, j in T.grid(4,4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vj]
    

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
print("TVM add test passed!")



# torch version conv
import torch

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch

@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv(data: T.Buffer((N, CI, H, W), "int64"),
           weight: T.Buffer((CO, CI, K, K), "int64"),
           result: T.Buffer((N, CO, OUT_H, OUT_W), "int64")):
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    for n, co, oh, ow in T.grid(N, CO, OUT_H, OUT_W):
      with T.block("init"):
        vn = T.axis.spatial(N, n)
        vco = T.axis.spatial(CO, co)
        voh = T.axis.spatial(OUT_H, oh)
        vow = T.axis.spatial(OUT_W, ow)
        result[vn, vco, voh, vow] = 0
    for n, co, oh, ow, ci, kh, kw in T.grid(N, CO, OUT_H, OUT_W, CI, K, K):
      with T.block("result"):
        vn = T.axis.spatial(N, n)
        vco = T.axis.spatial(CO, co)
        voh = T.axis.spatial(OUT_H, oh)
        vow = T.axis.spatial(OUT_W, ow)
        vci = T.axis.spatial(CI, ci)
        vkh = T.axis.spatial(K, kh)
        vkw = T.axis.spatial(K, kw)
        result[vn, vco, voh, vow] += data[vn, vci, voh + vkh, vow + vkw] * \
                                                  weight[vco, vci, vkh, vkw]

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
print("TVM conv test passed!")


def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
         

@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu(A: T.Buffer((16, 128, 128), "float32"),
               B: T.Buffer((16, 128, 128), "float32"),
               C: T.Buffer((16, 128, 128), "float32")):
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    Y = T.alloc_buffer([16, 128, 128], dtype="float32")
    for n, i, j, k in T.grid(16, 128, 128, 128):
       with T.block("Y"):
        vn = T.axis.spatial(16, n)
        vi = T.axis.spatial(128, i)
        vj = T.axis.spatial(128, j)
        vk = T.axis.reduce(128, k)
        with T.init():
          Y[vn, vi, vj] = 0
        Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi, vk] * B[vn, vk, vj]
    for n, i, j in T.grid(16, 128, 128):
      with T.block("C"):
        vn = T.axis.spatial(16, n)
        vi = T.axis.spatial(128, i)
        vj = T.axis.spatial(128, j)
        C[vn, vi, vj] = T.max(Y[vn, vi, vj], 0)

sch = tvm.tir.Schedule(MyBmmRelu)
# Step 0. Set tile size
L1 = 16
L2 = 8
# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
C = sch.get_block("C", func_name="bmm_relu")
# Step 2. Get loops
n, i, j, k = sch.get_loops(Y)
cn, ci, cj = sch.get_loops(C)
# Step 3. Organize loops
j0, j1 = sch.split(j, factors=[L1,L2])
cj0, cj1 = sch.split(cj, factors=[L1,L2])
sch.compute_at(Y, cj0)
sch.vectorize(cj1)
Y = sch.get_block("Y", func_name="bmm_relu")
n, i, j0, j1, k = sch.get_loops(Y)
sch.parallel(n)
# Step 4. Decompose reduction
Y_init = sch.decompose_reduction(Y, j1)
n, i, j0, ax0_init = sch.get_loops(Y_init)
sch.vectorize(ax0_init)
Y_update = sch.get_block("Y_update", func_name="bmm_relu")
n, i, j_0, ax0, ax1 = sch.get_loops(Y_update)
ax1_0, ax1_1 = sch.split(ax1, factors=[L1,L2])
sch.reorder(ax1_0, ax1_1, ax0)
sch.unroll(ax1_1)


print(sch.mod.script())
# tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")

before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))
f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
