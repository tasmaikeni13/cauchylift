/**
 * CauchyLift: Native ROCm/HIP Fused Optimizer Kernel
 *
 * Implements the fused CauchyLift optimizer step:
 * 1. Historical momentum buffer update:
 *    M_t = beta * M_{t-1} + (1 - beta) * G_t
 * 2. Additive fiber RMS energy reduction:
 *    row_rms[i] = sqrt(sum_j M_{ij}^2 / n)
 *    col_rms[j] = sqrt(sum_i M_{ij}^2 / m)
 *    D_{ij} = row_rms[i] + col_rms[j]
 * 3. Direction normalization:
 *    Z_{ij} = M_{ij} / D_{ij}
 *    U = sqrt(max(m, n)) * Z / ||Z||_F
 * 4. Decoupled weight decay and parameter update:
 *    W_{t+1} = W_t * (1 - lr * weight_decay) - lr * U
 *
 * Supports both single-tensor and multi-tensor foreach execution in FP32 and BF16.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kElementThreads = 512;
constexpr int kMetaFields = 9;

enum MetaField : int {
  kParameterPointer = 0,
  kGradientPointer = 1,
  kMomentumPointer = 2,
  kRows = 3,
  kColumns = 4,
  kRowOffset = 5,
  kColumnOffset = 6,
  kTileOffset = 7,
  kTileCount = 8,
};

__device__ __forceinline__ int locate_metadata(
    const int64_t* metadata,
    int tensor_count,
    int64_t work_index,
    int offset_field) {
  int lower = 0;
  int upper = tensor_count;
  while (lower + 1 < upper) {
    int middle = (lower + upper) / 2;
    if (metadata[middle * kMetaFields + offset_field] <= work_index) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

// -------------------------------------------------------------
// Single Tensor Kernels
// -------------------------------------------------------------

template <typename scalar_t>
__global__ void momentum_and_fiber_energy_kernel(
    const scalar_t* gradient,
    scalar_t* momentum_buffer,
    float* row_energy,
    float* column_energy,
    int64_t size,
    int64_t rows,
    int64_t columns,
    float momentum_beta) {
  float one_minus_beta = 1.0f - momentum_beta;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float g = static_cast<float>(gradient[idx]);
    float m = static_cast<float>(momentum_buffer[idx]);
    // M_t = beta * M_{t-1} + (1 - beta) * G_t
    float new_m = momentum_beta * m + one_minus_beta * g;
    momentum_buffer[idx] = static_cast<scalar_t>(new_m);

    float sq = new_m * new_m;
    int64_t r = idx / columns;
    int64_t c = idx % columns;
    atomicAdd(&row_energy[r], sq);
    atomicAdd(&column_energy[c], sq);
  }
}

__global__ void finalize_rms_kernel(
    float* row_energy,
    int64_t rows,
    int64_t columns,
    float* column_energy) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
       r < rows;
       r += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    row_energy[r] = sqrtf(row_energy[r] / static_cast<float>(columns));
  }
  for (int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
       c < columns;
       c += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    column_energy[c] = sqrtf(column_energy[c] / static_cast<float>(rows));
  }
}

template <typename scalar_t>
__global__ void compute_raw_norm_kernel(
    const scalar_t* momentum_buffer,
    const float* row_rms,
    const float* col_rms,
    float* norm_square,
    int64_t size,
    int64_t columns) {
  float local_sum = 0.0f;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float m = static_cast<float>(momentum_buffer[idx]);
    if (m != 0.0f) {
      int64_t r = idx / columns;
      int64_t c = idx % columns;
      float denom = row_rms[r] + col_rms[c];
      if (denom > 0.0f) {
        float val = m / denom;
        local_sum += val * val;
      }
    }
  }

  // Block reduction
  __shared__ float sdata[kThreads];
  int tid = threadIdx.x;
  sdata[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(norm_square, sdata[0]);
  }
}

template <typename scalar_t>
__global__ void cauchylift_update_kernel(
    scalar_t* parameter,
    const scalar_t* momentum_buffer,
    const float* row_rms,
    const float* col_rms,
    const float* norm_square,
    int64_t size,
    int64_t rows,
    int64_t columns,
    float learning_rate,
    float weight_decay) {
  float raw_norm = sqrtf(*norm_square);
  float radius = sqrtf(static_cast<float>(max(rows, columns)));
  float scale = (raw_norm > 0.0f) ? (radius / raw_norm) : 0.0f;
  float decay_factor = 1.0f - learning_rate * weight_decay;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float p = static_cast<float>(parameter[idx]);
    // 1. Decoupled weight decay
    if (weight_decay != 0.0f) {
      p *= decay_factor;
    }

    // 2. CauchyLift step
    float m = static_cast<float>(momentum_buffer[idx]);
    if (m != 0.0f && scale > 0.0f) {
      int64_t r = idx / columns;
      int64_t c = idx % columns;
      float denom = row_rms[r] + col_rms[c];
      if (denom > 0.0f) {
        float dir = (m / denom) * scale;
        p -= learning_rate * dir;
      }
    }
    parameter[idx] = static_cast<scalar_t>(p);
  }
}

// -------------------------------------------------------------
// Multi-Tensor Foreach Kernels
// -------------------------------------------------------------

__global__ void foreach_init_zero_kernel(
    float* row_energy,
    int64_t total_rows,
    float* column_energy,
    int64_t total_columns,
    float* norm_square,
    int tensor_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_rows) {
    row_energy[idx] = 0.0f;
  }
  if (idx < total_columns) {
    column_energy[idx] = 0.0f;
  }
  if (idx < tensor_count) {
    norm_square[idx] = 0.0f;
  }
}

template <typename scalar_t>
__global__ void foreach_momentum_and_fiber_energy_kernel(
    const int64_t* metadata,
    int tensor_count,
    float* row_energy,
    float* column_energy,
    float momentum_beta) {
  int64_t tile_index = blockIdx.x;
  int tensor_index = locate_metadata(metadata, tensor_count, tile_index, kTileOffset);
  const int64_t* meta = metadata + tensor_index * kMetaFields;

  auto gradient = reinterpret_cast<const scalar_t*>(meta[kGradientPointer]);
  auto momentum_buffer = reinterpret_cast<scalar_t*>(meta[kMomentumPointer]);
  int64_t rows = meta[kRows];
  int64_t columns = meta[kColumns];
  int64_t row_offset = meta[kRowOffset];
  int64_t col_offset = meta[kColumnOffset];
  int64_t tile_offset = meta[kTileOffset];
  int64_t local_tile = tile_index - tile_offset;

  int64_t total_elements = rows * columns;
  int64_t start_idx = local_tile * kElementThreads + threadIdx.x;
  float one_minus_beta = 1.0f - momentum_beta;

  if (start_idx < total_elements) {
    float g = static_cast<float>(gradient[start_idx]);
    float m = static_cast<float>(momentum_buffer[start_idx]);
    float new_m = momentum_beta * m + one_minus_beta * g;
    momentum_buffer[start_idx] = static_cast<scalar_t>(new_m);

    float sq = new_m * new_m;
    int64_t r = start_idx / columns;
    int64_t c = start_idx % columns;
    atomicAdd(&row_energy[row_offset + r], sq);
    atomicAdd(&column_energy[col_offset + c], sq);
  }
}

template <typename scalar_t>
__global__ void foreach_norm_kernel(
    const int64_t* metadata,
    int tensor_count,
    const float* row_energy,
    const float* column_energy,
    float* norm_square) {
  int64_t tile_index = blockIdx.x;
  int tensor_index = locate_metadata(metadata, tensor_count, tile_index, kTileOffset);
  const int64_t* meta = metadata + tensor_index * kMetaFields;

  auto momentum_buffer = reinterpret_cast<const scalar_t*>(meta[kMomentumPointer]);
  int64_t rows = meta[kRows];
  int64_t columns = meta[kColumns];
  int64_t row_offset = meta[kRowOffset];
  int64_t col_offset = meta[kColumnOffset];
  int64_t tile_offset = meta[kTileOffset];
  int64_t local_tile = tile_index - tile_offset;

  int64_t total_elements = rows * columns;
  int64_t idx = local_tile * kElementThreads + threadIdx.x;

  float local_val = 0.0f;
  if (idx < total_elements) {
    float m = static_cast<float>(momentum_buffer[idx]);
    if (m != 0.0f) {
      int64_t r = idx / columns;
      int64_t c = idx % columns;
      float row_rms = sqrtf(row_energy[row_offset + r] / static_cast<float>(columns));
      float col_rms = sqrtf(column_energy[col_offset + c] / static_cast<float>(rows));
      float denom = row_rms + col_rms;
      if (denom > 0.0f) {
        float raw = m / denom;
        local_val = raw * raw;
      }
    }
  }

  // Block reduction
  __shared__ float sdata[kElementThreads];
  int tid = threadIdx.x;
  sdata[tid] = local_val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && sdata[0] > 0.0f) {
    atomicAdd(&norm_square[tensor_index], sdata[0]);
  }
}

template <typename scalar_t>
__global__ void foreach_output_kernel(
    const int64_t* metadata,
    int tensor_count,
    const float* row_energy,
    const float* column_energy,
    const float* norm_square,
    float learning_rate,
    float weight_decay) {
  int64_t tile_index = blockIdx.x;
  int tensor_index = locate_metadata(metadata, tensor_count, tile_index, kTileOffset);
  const int64_t* meta = metadata + tensor_index * kMetaFields;

  auto parameter = reinterpret_cast<scalar_t*>(meta[kParameterPointer]);
  auto momentum_buffer = reinterpret_cast<const scalar_t*>(meta[kMomentumPointer]);
  int64_t rows = meta[kRows];
  int64_t columns = meta[kColumns];
  int64_t row_offset = meta[kRowOffset];
  int64_t col_offset = meta[kColumnOffset];
  int64_t tile_offset = meta[kTileOffset];
  int64_t local_tile = tile_index - tile_offset;

  int64_t total_elements = rows * columns;
  int64_t idx = local_tile * kElementThreads + threadIdx.x;

  if (idx < total_elements) {
    float raw_norm = sqrtf(norm_square[tensor_index]);
    float radius = sqrtf(static_cast<float>(max(rows, columns)));
    float scale = (raw_norm > 0.0f) ? (radius / raw_norm) : 0.0f;
    float decay_factor = 1.0f - learning_rate * weight_decay;

    float p = static_cast<float>(parameter[idx]);
    if (weight_decay != 0.0f) {
      p *= decay_factor;
    }

    float m = static_cast<float>(momentum_buffer[idx]);
    if (m != 0.0f && scale > 0.0f) {
      int64_t r = idx / columns;
      int64_t c = idx % columns;
      float row_rms = sqrtf(row_energy[row_offset + r] / static_cast<float>(columns));
      float col_rms = sqrtf(column_energy[col_offset + c] / static_cast<float>(rows));
      float denom = row_rms + col_rms;
      if (denom > 0.0f) {
        float dir = (m / denom) * scale;
        p -= learning_rate * dir;
      }
    }
    parameter[idx] = static_cast<scalar_t>(p);
  }
}

} // namespace

// -------------------------------------------------------------
// Host Dispatchers
// -------------------------------------------------------------

at::Tensor cauchylift_step_hip(
    at::Tensor& parameter,
    const at::Tensor& gradient,
    at::Tensor& momentum_buffer,
    double learning_rate,
    double momentum,
    double weight_decay) {
  TORCH_CHECK(parameter.is_cuda(), "parameter must be a ROCm/CUDA tensor");
  TORCH_CHECK(gradient.is_cuda(), "gradient must be a ROCm/CUDA tensor");
  TORCH_CHECK(momentum_buffer.is_cuda(), "momentum_buffer must be a ROCm/CUDA tensor");
  TORCH_CHECK(parameter.is_contiguous(), "parameter must be contiguous");
  TORCH_CHECK(gradient.is_contiguous(), "gradient must be contiguous");
  TORCH_CHECK(momentum_buffer.is_contiguous(), "momentum_buffer must be contiguous");

  c10::cuda::CUDAGuard guard(parameter.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t size = parameter.numel();
  int64_t rows = (parameter.dim() <= 1) ? 1 : parameter.size(0);
  int64_t columns = size / rows;

  auto opts_f32 = parameter.options().dtype(at::kFloat);
  at::Tensor row_energy = at::zeros({rows}, opts_f32);
  at::Tensor column_energy = at::zeros({columns}, opts_f32);
  at::Tensor norm_square = at::zeros({1}, opts_f32);

  int blocks = static_cast<int>(std::min<int64_t>((size + kThreads - 1) / kThreads, 65535));

  if (parameter.scalar_type() == at::kFloat) {
    momentum_and_fiber_energy_kernel<float><<<blocks, kThreads, 0, stream>>>(
        gradient.data_ptr<float>(),
        momentum_buffer.data_ptr<float>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        size, rows, columns,
        static_cast<float>(momentum));

    int rms_blocks = static_cast<int>(std::max<int64_t>((max(rows, columns) + kThreads - 1) / kThreads, 1));
    finalize_rms_kernel<<<rms_blocks, kThreads, 0, stream>>>(
        row_energy.data_ptr<float>(), rows, columns, column_energy.data_ptr<float>());

    compute_raw_norm_kernel<float><<<blocks, kThreads, 0, stream>>>(
        momentum_buffer.data_ptr<float>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        size, columns);

    cauchylift_update_kernel<float><<<blocks, kThreads, 0, stream>>>(
        parameter.data_ptr<float>(),
        momentum_buffer.data_ptr<float>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        size, rows, columns,
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay));
  } else if (parameter.scalar_type() == at::kBFloat16) {
    momentum_and_fiber_energy_kernel<at::BFloat16><<<blocks, kThreads, 0, stream>>>(
        gradient.data_ptr<at::BFloat16>(),
        momentum_buffer.data_ptr<at::BFloat16>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        size, rows, columns,
        static_cast<float>(momentum));

    int rms_blocks = static_cast<int>(std::max<int64_t>((max(rows, columns) + kThreads - 1) / kThreads, 1));
    finalize_rms_kernel<<<rms_blocks, kThreads, 0, stream>>>(
        row_energy.data_ptr<float>(), rows, columns, column_energy.data_ptr<float>());

    compute_raw_norm_kernel<at::BFloat16><<<blocks, kThreads, 0, stream>>>(
        momentum_buffer.data_ptr<at::BFloat16>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        size, columns);

    cauchylift_update_kernel<at::BFloat16><<<blocks, kThreads, 0, stream>>>(
        parameter.data_ptr<at::BFloat16>(),
        momentum_buffer.data_ptr<at::BFloat16>(),
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        size, rows, columns,
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay));
  } else {
    TORCH_CHECK(false, "cauchylift_step_hip only supports float32 and bfloat16");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return norm_square;
}

at::Tensor cauchylift_foreach_step_hip(
    const std::vector<at::Tensor>& parameters,
    const std::vector<at::Tensor>& gradients,
    const std::vector<at::Tensor>& momentum_buffers,
    double learning_rate,
    double momentum,
    double weight_decay) {
  int tensor_count = static_cast<int>(parameters.size());
  if (tensor_count == 0) {
    return at::empty({0}, at::kFloat);
  }

  const at::Tensor& first = parameters[0];
  c10::cuda::CUDAGuard guard(first.device());

  at::Tensor metadata_cpu = at::empty({tensor_count * kMetaFields}, at::kLong);
  int64_t* metadata_host = metadata_cpu.data_ptr<int64_t>();

  int64_t total_rows = 0;
  int64_t total_columns = 0;
  int64_t total_tiles = 0;

  for (int t = 0; t < tensor_count; ++t) {
    const at::Tensor& p = parameters[t];
    const at::Tensor& g = gradients[t];
    const at::Tensor& m = momentum_buffers[t];
    int64_t size = p.numel();
    int64_t r = (p.dim() <= 1) ? 1 : p.size(0);
    int64_t c = size / r;
    int64_t tiles = (size + kElementThreads - 1) / kElementThreads;

    metadata_host[t * kMetaFields + kParameterPointer] = reinterpret_cast<int64_t>(p.data_ptr());
    metadata_host[t * kMetaFields + kGradientPointer] = reinterpret_cast<int64_t>(g.data_ptr());
    metadata_host[t * kMetaFields + kMomentumPointer] = reinterpret_cast<int64_t>(m.data_ptr());
    metadata_host[t * kMetaFields + kRows] = r;
    metadata_host[t * kMetaFields + kColumns] = c;
    metadata_host[t * kMetaFields + kRowOffset] = total_rows;
    metadata_host[t * kMetaFields + kColumnOffset] = total_columns;
    metadata_host[t * kMetaFields + kTileOffset] = total_tiles;
    metadata_host[t * kMetaFields + kTileCount] = tiles;

    total_rows += r;
    total_columns += c;
    total_tiles += tiles;
  }

  at::Tensor metadata = metadata_cpu.to(first.device(), /*non_blocking=*/true);
  auto opts_f32 = first.options().dtype(at::kFloat);
  at::Tensor row_energy = at::empty({total_rows}, opts_f32);
  at::Tensor column_energy = at::empty({total_columns}, opts_f32);
  at::Tensor norm_square = at::empty({tensor_count}, opts_f32);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 1. Initialize workspaces to 0
  foreach_init_zero_kernel<<<static_cast<int>(std::min<int64_t>((total_rows + total_columns + kThreads - 1) / kThreads, 4096)), kThreads, 0, stream>>>(
      row_energy.data_ptr<float>(), total_rows,
      column_energy.data_ptr<float>(), total_columns,
      norm_square.data_ptr<float>(), tensor_count);

  if (first.scalar_type() == at::kFloat) {
    foreach_momentum_and_fiber_energy_kernel<float><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        static_cast<float>(momentum));

    foreach_norm_kernel<float><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>());

    foreach_output_kernel<float><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay));
  } else {
    foreach_momentum_and_fiber_energy_kernel<at::BFloat16><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        static_cast<float>(momentum));

    foreach_norm_kernel<at::BFloat16><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>());

    foreach_output_kernel<at::BFloat16><<<total_tiles, kElementThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        tensor_count,
        row_energy.data_ptr<float>(),
        column_energy.data_ptr<float>(),
        norm_square.data_ptr<float>(),
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return norm_square;
}

TORCH_LIBRARY(cauchylift_native, module) {
  module.def("step_(Tensor(a!) parameter, Tensor gradient, Tensor(b!) momentum_buffer, float learning_rate, float momentum, float weight_decay) -> Tensor");
  module.def("foreach_step_(Tensor(a!)[] parameters, Tensor[] gradients, Tensor(b!)[] momentum_buffers, float learning_rate, float momentum, float weight_decay) -> Tensor");
  module.impl("step_", &cauchylift_step_hip);
  module.impl("foreach_step_", &cauchylift_foreach_step_hip);
}
