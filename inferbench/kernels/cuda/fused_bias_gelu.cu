// Fused bias-add + GELU CUDA kernel.
//
// Computes  y = gelu(x + bias)  in a single pass over the data, where bias is
// broadcast over the leading (row) dimension. This is the pointwise epilogue of
// a transformer MLP's first linear layer; fusing the bias-add with the GELU
// avoids materializing the intermediate (x + bias) tensor in HBM — the op is
// memory-bound, so removing one full read+write of the activation is the win.
//
// GELU uses the tanh approximation (matches nn.GELU(approximate="tanh") and the
// GPT/LLaMA-family MLPs), evaluated in fp32 regardless of IO dtype for accuracy.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluCoeff = 0.044715f;

template <typename scalar_t>
__global__ void fused_bias_gelu_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    const int64_t n_cols,
    const int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < numel; i += stride) {
    const int64_t col = i % n_cols;  // bias is broadcast over rows
    const float v = static_cast<float>(x[i]) + static_cast<float>(bias[col]);
    const float inner = kSqrt2OverPi * (v + kGeluCoeff * v * v * v);
    const float out = 0.5f * v * (1.0f + tanhf(inner));
    y[i] = static_cast<scalar_t>(out);
  }
}

}  // namespace

torch::Tensor fused_bias_gelu(torch::Tensor x, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
  TORCH_CHECK(x.size(-1) == bias.size(0), "bias size must match x's last dim");

  auto x_c = x.contiguous();
  auto bias_c = bias.contiguous();
  auto y = torch::empty_like(x_c);

  const int64_t numel = x_c.numel();
  const int64_t n_cols = bias_c.size(0);
  const int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  // Cap grid to a sane size; the grid-stride loop covers the remainder.
  const int max_blocks = 65535;
  const int grid = blocks < max_blocks ? blocks : max_blocks;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_c.scalar_type(), "fused_bias_gelu", ([&] {
        fused_bias_gelu_kernel<scalar_t><<<grid, threads>>>(
            x_c.data_ptr<scalar_t>(),
            bias_c.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            n_cols,
            numel);
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_bias_gelu", &fused_bias_gelu, "Fused bias-add + GELU (CUDA)");
}
