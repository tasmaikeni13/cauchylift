#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int positive_float_bits(float value) {
  return __float_as_int(value);
}

__device__ __forceinline__ float bits_float(int value) {
  return __int_as_float(value);
}

template <typename scalar_t>
__global__ void analyze_partial_kernel(
    const scalar_t* gradient, int64_t size, int* partials) {
  __shared__ int maxima[kThreads];
  __shared__ int counts[kThreads];
  __shared__ int nonfinite[kThreads];
  int local_maximum = 0;
  int local_count = 0;
  int local_nonfinite = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < size;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = static_cast<float>(gradient[index]);
    if (!isfinite(value)) {
      local_nonfinite = 1;
    } else {
      float magnitude = fabsf(value);
      if (magnitude > 0.0f) {
        ++local_count;
        local_maximum = max(local_maximum, positive_float_bits(magnitude));
      }
    }
  }
  maxima[threadIdx.x] = local_maximum;
  counts[threadIdx.x] = local_count;
  nonfinite[threadIdx.x] = local_nonfinite;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
      counts[threadIdx.x] += counts[threadIdx.x + stride];
      nonfinite[threadIdx.x] |= nonfinite[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[3 * blockIdx.x] = maxima[0];
    partials[3 * blockIdx.x + 1] = counts[0];
    partials[3 * blockIdx.x + 2] = nonfinite[0];
  }
}

__global__ void analyze_finalize_kernel(
    const int* partials, int block_count, int* status) {
  __shared__ int maxima[kThreads];
  __shared__ int counts[kThreads];
  __shared__ int nonfinite[kThreads];
  int local_maximum = 0;
  int local_count = 0;
  int local_nonfinite = 0;
  for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
    local_maximum = max(local_maximum, partials[3 * index]);
    local_count += partials[3 * index + 1];
    local_nonfinite |= partials[3 * index + 2];
  }
  maxima[threadIdx.x] = local_maximum;
  counts[threadIdx.x] = local_count;
  nonfinite[threadIdx.x] = local_nonfinite;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
      counts[threadIdx.x] += counts[threadIdx.x + stride];
      nonfinite[threadIdx.x] |= nonfinite[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    status[0] = maxima[0];
    status[1] = counts[0];
    status[2] = nonfinite[0];
    status[3] = 0;
  }
}

template <typename scalar_t>
__global__ void marginal_energy_tiled_kernel(
    const scalar_t* gradient,
    int64_t rows,
    int64_t columns,
    const int* status,
    float* row_energy,
    float* column_energy) {
  if (status[2] || status[0] == 0) return;
  __shared__ float tile[16][17];
  int64_t row = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.y;
  int64_t column = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.x;
  float maximum = bits_float(status[0]);
  float square = 0.0f;
  if (row < rows && column < columns) {
    float value = static_cast<float>(gradient[row * columns + column]) / maximum;
    square = value * value;
  }
  tile[threadIdx.y][threadIdx.x] = square;
  __syncthreads();
  if (threadIdx.x == 0 && row < rows) {
    float sum = 0.0f;
    for (int x = 0; x < 16; ++x) sum += tile[threadIdx.y][x];
    atomicAdd(row_energy + row, sum);
  }
  if (threadIdx.y == 0 && column < columns) {
    float sum = 0.0f;
    for (int y = 0; y < 16; ++y) sum += tile[y][threadIdx.x];
    atomicAdd(column_energy + column, sum);
  }
}

__global__ void exclusion_scan_kernel(
    const float* row_energy,
    int64_t rows,
    float* outside_rows,
    const float* column_energy,
    int64_t columns,
    float* outside_columns) {
  const float* input = blockIdx.x == 0 ? row_energy : column_energy;
  float* output = blockIdx.x == 0 ? outside_rows : outside_columns;
  int64_t size = blockIdx.x == 0 ? rows : columns;
  __shared__ float segment_sums[kThreads];
  int64_t segment = (size + blockDim.x - 1) / blockDim.x;
  int64_t begin = min<int64_t>(size, threadIdx.x * segment);
  int64_t end = min<int64_t>(size, begin + segment);
  float local_total = 0.0f;
  for (int64_t index = begin; index < end; ++index) local_total += input[index];
  segment_sums[threadIdx.x] = local_total;
  __syncthreads();
  float prefix = 0.0f;
  float suffix = 0.0f;
  for (int thread = 0; thread < threadIdx.x; ++thread) prefix += segment_sums[thread];
  for (int thread = threadIdx.x + 1; thread < blockDim.x; ++thread)
    suffix += segment_sums[thread];
  for (int64_t index = begin; index < end; ++index) {
    output[index] = prefix;
    prefix += input[index];
  }
  for (int64_t index = end; index-- > begin;) {
    output[index] += suffix;
    suffix += input[index];
  }
}

template <typename scalar_t>
__global__ void denominator_min_kernel(
    const scalar_t* gradient,
    int64_t size,
    int64_t columns,
    const int* status,
    const float* outside_rows,
    const float* outside_columns,
    int* partials) {
  if (status[2] || status[1] <= 1) return;
  __shared__ int minima[kThreads];
  __shared__ int invalid[kThreads];
  int local_minimum = std::numeric_limits<int>::max();
  int local_invalid = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < size;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = static_cast<float>(gradient[index]);
    if (value == 0.0f) continue;
    float denominator = outside_rows[index / columns] + outside_columns[index % columns];
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
      local_invalid = 1;
    } else {
      local_minimum = min(local_minimum, positive_float_bits(denominator));
    }
  }
  minima[threadIdx.x] = local_minimum;
  invalid[threadIdx.x] = local_invalid;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      minima[threadIdx.x] = min(minima[threadIdx.x], minima[threadIdx.x + stride]);
      invalid[threadIdx.x] |= invalid[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[2 * blockIdx.x] = minima[0];
    partials[2 * blockIdx.x + 1] = invalid[0];
  }
}

__global__ void denominator_finalize_kernel(
    const int* partials,
    int block_count,
    int* minimum_bits,
    int* status) {
  if (status[2] || status[1] <= 1) {
    if (threadIdx.x == 0) {
      minimum_bits[0] = std::numeric_limits<int>::max();
      status[3] = 0;
    }
    return;
  }
  __shared__ int minima[kThreads];
  __shared__ int invalid[kThreads];
  int local_minimum = std::numeric_limits<int>::max();
  int local_invalid = 0;
  for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
    local_minimum = min(local_minimum, partials[2 * index]);
    local_invalid |= partials[2 * index + 1];
  }
  minima[threadIdx.x] = local_minimum;
  invalid[threadIdx.x] = local_invalid;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      minima[threadIdx.x] = min(minima[threadIdx.x], minima[threadIdx.x + stride]);
      invalid[threadIdx.x] |= invalid[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    minimum_bits[0] = minima[0];
    status[3] = invalid[0];
  }
}

template <typename scalar_t>
__global__ void raw_norm_kernel(
    const scalar_t* gradient,
    int64_t size,
    int64_t columns,
    const int* status,
    const float* outside_rows,
    const float* outside_columns,
    const int* minimum_bits,
    float* partials) {
  if (status[2] || status[3] || status[1] <= 1) return;
  __shared__ float sums[kThreads];
  float maximum = bits_float(status[0]);
  float minimum = bits_float(minimum_bits[0]);
  float local_sum = 0.0f;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < size;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = static_cast<float>(gradient[index]);
    if (value == 0.0f) continue;
    float denominator = outside_rows[index / columns] + outside_columns[index % columns];
    float raw = (value / maximum) * minimum / denominator;
    local_sum += raw * raw;
  }
  sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sums[0];
}

__global__ void raw_norm_finalize_kernel(
    const float* partials, int block_count, const int* status, float* norm_square) {
  if (status[2] || status[3] || status[1] <= 1) {
    if (threadIdx.x == 0) norm_square[0] = 0.0f;
    return;
  }
  __shared__ float sums[kThreads];
  float local_sum = 0.0f;
  for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
    local_sum += partials[index];
  }
  sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) norm_square[0] = sums[0];
}

template <typename scalar_t, bool Update>
__global__ void output_kernel(
    scalar_t* parameter,
    const scalar_t* gradient,
    float* output,
    int64_t size,
    int64_t columns,
    int64_t minimum_dimension,
    const int* status,
    const float* outside_rows,
    const float* outside_columns,
    const int* minimum_bits,
    const float* norm_square,
    float learning_rate) {
  float maximum = bits_float(status[0]);
  float minimum = bits_float(minimum_bits[0]);
  float radius = sqrtf(static_cast<float>(minimum_dimension));
  float norm = sqrtf(norm_square[0]);
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < size;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = static_cast<float>(gradient[index]);
    float direction = 0.0f;
    if (!status[2] && !status[3]) {
      if (status[1] == 1 && value != 0.0f) {
        direction = copysignf(radius, value);
      } else if (status[1] > 1 && value != 0.0f && norm > 0.0f) {
        float denominator =
            outside_rows[index / columns] + outside_columns[index % columns];
        float raw = (value / maximum) * minimum / denominator;
        direction = radius * raw / norm;
      }
    }
    if constexpr (Update) {
      if (!status[2] && !status[3]) {
        float current = static_cast<float>(parameter[index]);
        parameter[index] = static_cast<scalar_t>(current - learning_rate * direction);
      }
    } else {
      output[index] = direction;
    }
  }
}

struct Workspace {
  at::Tensor status;
  at::Tensor row_energy;
  at::Tensor column_energy;
  at::Tensor outside_rows;
  at::Tensor outside_columns;
  at::Tensor minimum_bits;
  at::Tensor norm_square;
  at::Tensor analysis_partials;
  at::Tensor denominator_partials;
  at::Tensor norm_partials;
};

Workspace make_workspace(
    const at::Tensor& gradient, int64_t rows, int64_t columns, int blocks) {
  auto float_options = gradient.options().dtype(at::kFloat);
  auto int_options = gradient.options().dtype(at::kInt);
  return {
      at::empty({4}, int_options),
      at::zeros({rows}, float_options),
      at::zeros({columns}, float_options),
      at::empty({rows}, float_options),
      at::empty({columns}, float_options),
      at::empty({1}, int_options),
      at::empty({1}, float_options),
      at::empty({blocks * 3}, int_options),
      at::empty({blocks * 2}, int_options),
      at::empty({blocks}, float_options),
  };
}

template <typename scalar_t>
void launch_common(
    const at::Tensor& gradient,
    int64_t rows,
    int64_t columns,
    Workspace& workspace,
    cudaStream_t stream) {
  int64_t size = gradient.numel();
  int blocks = static_cast<int>(std::min<int64_t>((size + kThreads - 1) / kThreads, 4096));
  analyze_partial_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
      gradient.data_ptr<scalar_t>(), size, workspace.analysis_partials.data_ptr<int>());
  analyze_finalize_kernel<<<1, kThreads, 0, stream>>>(
      workspace.analysis_partials.data_ptr<int>(), blocks, workspace.status.data_ptr<int>());
  dim3 marginal_threads(16, 16);
  dim3 marginal_blocks(
      static_cast<unsigned int>((columns + 15) / 16),
      static_cast<unsigned int>((rows + 15) / 16));
  marginal_energy_tiled_kernel<scalar_t><<<marginal_blocks, marginal_threads, 0, stream>>>(
      gradient.data_ptr<scalar_t>(),
      rows,
      columns,
      workspace.status.data_ptr<int>(),
      workspace.row_energy.data_ptr<float>(),
      workspace.column_energy.data_ptr<float>());
  exclusion_scan_kernel<<<2, kThreads, 0, stream>>>(
      workspace.row_energy.data_ptr<float>(),
      rows,
      workspace.outside_rows.data_ptr<float>(),
      workspace.column_energy.data_ptr<float>(),
      columns,
      workspace.outside_columns.data_ptr<float>());
  denominator_min_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
      gradient.data_ptr<scalar_t>(),
      size,
      columns,
      workspace.status.data_ptr<int>(),
      workspace.outside_rows.data_ptr<float>(),
      workspace.outside_columns.data_ptr<float>(),
      workspace.denominator_partials.data_ptr<int>());
  denominator_finalize_kernel<<<1, kThreads, 0, stream>>>(
      workspace.denominator_partials.data_ptr<int>(),
      blocks,
      workspace.minimum_bits.data_ptr<int>(),
      workspace.status.data_ptr<int>());
  raw_norm_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
      gradient.data_ptr<scalar_t>(),
      size,
      columns,
      workspace.status.data_ptr<int>(),
      workspace.outside_rows.data_ptr<float>(),
      workspace.outside_columns.data_ptr<float>(),
      workspace.minimum_bits.data_ptr<int>(),
      workspace.norm_partials.data_ptr<float>());
  raw_norm_finalize_kernel<<<1, kThreads, 0, stream>>>(
      workspace.norm_partials.data_ptr<float>(),
      blocks,
      workspace.status.data_ptr<int>(),
      workspace.norm_square.data_ptr<float>());
}

void check_input(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "CauchyLift HIP input must be on a ROCm device");
  TORCH_CHECK(tensor.is_contiguous(), "CauchyLift HIP input must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat ||
          tensor.scalar_type() == at::kBFloat16,
      "CauchyLift HIP supports FP32 and BF16");
  TORCH_CHECK(tensor.numel() > 0, "CauchyLift does not support empty parameters");
}

std::pair<int64_t, int64_t> matrix_shape(const at::Tensor& tensor) {
  if (tensor.dim() == 0) return {1, 1};
  if (tensor.dim() == 1) return {tensor.numel(), 1};
  int64_t rows = tensor.size(0);
  return {rows, tensor.numel() / rows};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> cauchylift_direction_hip(at::Tensor gradient) {
  check_input(gradient);
  c10::cuda::CUDAGuard guard(gradient.device());
  auto [rows, columns] = matrix_shape(gradient);
  int64_t size = gradient.numel();
  int blocks = static_cast<int>(std::min<int64_t>((size + kThreads - 1) / kThreads, 4096));
  Workspace workspace = make_workspace(gradient, rows, columns, blocks);
  at::Tensor output = at::empty_like(gradient, gradient.options().dtype(at::kFloat));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (gradient.scalar_type() == at::kFloat) {
    launch_common<float>(gradient, rows, columns, workspace, stream);
    output_kernel<float, false><<<blocks, kThreads, 0, stream>>>(
        nullptr,
        gradient.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        columns,
        std::min(rows, columns),
        workspace.status.data_ptr<int>(),
        workspace.outside_rows.data_ptr<float>(),
        workspace.outside_columns.data_ptr<float>(),
        workspace.minimum_bits.data_ptr<int>(),
        workspace.norm_square.data_ptr<float>(),
        0.0f);
  } else {
    launch_common<at::BFloat16>(gradient, rows, columns, workspace, stream);
    output_kernel<at::BFloat16, false><<<blocks, kThreads, 0, stream>>>(
        nullptr,
        gradient.data_ptr<at::BFloat16>(),
        output.data_ptr<float>(),
        size,
        columns,
        std::min(rows, columns),
        workspace.status.data_ptr<int>(),
        workspace.outside_rows.data_ptr<float>(),
        workspace.outside_columns.data_ptr<float>(),
        workspace.minimum_bits.data_ptr<int>(),
        workspace.norm_square.data_ptr<float>(),
        0.0f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output, workspace.status);
}

at::Tensor cauchylift_step_hip(
    at::Tensor parameter, at::Tensor gradient, double learning_rate) {
  check_input(parameter);
  check_input(gradient);
  TORCH_CHECK(parameter.sizes() == gradient.sizes(), "parameter and gradient shapes differ");
  TORCH_CHECK(parameter.scalar_type() == gradient.scalar_type(), "parameter and gradient dtypes differ");
  TORCH_CHECK(parameter.device() == gradient.device(), "parameter and gradient devices differ");
  c10::cuda::CUDAGuard guard(gradient.device());
  auto [rows, columns] = matrix_shape(gradient);
  int64_t size = gradient.numel();
  int blocks = static_cast<int>(std::min<int64_t>((size + kThreads - 1) / kThreads, 4096));
  Workspace workspace = make_workspace(gradient, rows, columns, blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (gradient.scalar_type() == at::kFloat) {
    launch_common<float>(gradient, rows, columns, workspace, stream);
    output_kernel<float, true><<<blocks, kThreads, 0, stream>>>(
        parameter.data_ptr<float>(),
        gradient.data_ptr<float>(),
        nullptr,
        size,
        columns,
        std::min(rows, columns),
        workspace.status.data_ptr<int>(),
        workspace.outside_rows.data_ptr<float>(),
        workspace.outside_columns.data_ptr<float>(),
        workspace.minimum_bits.data_ptr<int>(),
        workspace.norm_square.data_ptr<float>(),
        static_cast<float>(learning_rate));
  } else {
    launch_common<at::BFloat16>(gradient, rows, columns, workspace, stream);
    output_kernel<at::BFloat16, true><<<blocks, kThreads, 0, stream>>>(
        parameter.data_ptr<at::BFloat16>(),
        gradient.data_ptr<at::BFloat16>(),
        nullptr,
        size,
        columns,
        std::min(rows, columns),
        workspace.status.data_ptr<int>(),
        workspace.outside_rows.data_ptr<float>(),
        workspace.outside_columns.data_ptr<float>(),
        workspace.minimum_bits.data_ptr<int>(),
        workspace.norm_square.data_ptr<float>(),
        static_cast<float>(learning_rate));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return workspace.status;
}

TORCH_LIBRARY(cauchylift_native, module) {
  module.def("direction(Tensor gradient) -> (Tensor, Tensor)");
  module.def("step_(Tensor(a!) parameter, Tensor gradient, float learning_rate) -> Tensor");
}

TORCH_LIBRARY_IMPL(cauchylift_native, CUDA, module) {
  module.impl("direction", &cauchylift_direction_hip);
  module.impl("step_", &cauchylift_step_hip);
}
