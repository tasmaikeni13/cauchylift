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
constexpr int kElementThreads = 512;
constexpr int kMetaFields = 8;

enum MetaField : int {
  kParameterPointer = 0,
  kGradientPointer = 1,
  kRows = 2,
  kColumns = 3,
  kRowOffset = 4,
  kColumnOffset = 5,
  kTileOffset = 6,
  kTileCount = 7,
};

__device__ __forceinline__ int positive_float_bits(float value) {
  return __float_as_int(value);
}

__device__ __forceinline__ float bits_float(int value) {
  return __int_as_float(value);
}

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

__global__ void foreach_initialize_kernel(
    int* status,
    float* norm_square,
    float* scale,
    float* total_energy,
    float* row_energy,
    int64_t total_rows,
    float* column_energy,
    int64_t total_columns,
    int tensor_count,
    bool prevalidated_dense) {
  int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t global_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int index = global_thread; index < tensor_count; index += global_stride) {
    status[index * 4] = 0;
    status[index * 4 + 1] = prevalidated_dense ? 2 : 0;
    status[index * 4 + 2] = 0;
    status[index * 4 + 3] = 0;
    norm_square[index] = 0.0f;
    total_energy[index] = 0.0f;
    if (prevalidated_dense) {
      scale[index] = 1.0f;
    }
  }
  for (int64_t index = global_thread; index < total_rows; index += global_stride) {
    row_energy[index] = 0.0f;
  }
  for (int64_t index = global_thread; index < total_columns; index += global_stride) {
    column_energy[index] = 0.0f;
  }
}

__global__ void foreach_inverse_maximum_kernel(
    const int* status, float* inverse_maximum, int tensor_count) {
  for (int tensor = threadIdx.x; tensor < tensor_count; tensor += blockDim.x) {
    float maximum = bits_float(status[tensor * 4]);
    inverse_maximum[tensor] = maximum > 0.0f ? 1.0f / maximum : 0.0f;
  }
}

__global__ void foreach_finalize_scale_kernel(
    const int64_t* metadata,
    int* status,
    float* scale,
    const float* norm_square,
    int tensor_count) {
  for (int tensor = threadIdx.x; tensor < tensor_count; tensor += blockDim.x) {
    if (status[tensor * 4 + 1] <= 1 || status[tensor * 4 + 2] ||
        status[tensor * 4 + 3]) {
      continue;
    }
    float norm = sqrtf(norm_square[tensor]);
    if (!(norm > 0.0f) || !isfinite(norm)) {
      status[tensor * 4 + 3] = 1;
      continue;
    }
    const int64_t* item = metadata + tensor * kMetaFields;
    float radius = sqrtf(static_cast<float>(max(item[kRows], item[kColumns])));
    scale[tensor] = __fdividef(scale[tensor] * radius, norm);
  }
}

template <typename scalar_t, bool Analyze>
__global__ void foreach_marginal_energy_kernel(
    const int64_t* metadata,
    int tensor_count,
    int* analysis_partials,
    float* total_energy,
    float* row_energy,
    float* column_energy) {
  int tensor = locate_metadata(metadata, tensor_count, blockIdx.x, kTileOffset);
  const int64_t* item = metadata + tensor * kMetaFields;
  int64_t local_tile = blockIdx.x - item[kTileOffset];
  int64_t rows = item[kRows];
  int64_t columns = item[kColumns];
  int64_t tile_columns = (columns + 63) / 64;
  int64_t tile_row = local_tile / tile_columns;
  int64_t tile_column = local_tile % tile_columns;
  const scalar_t* gradient = reinterpret_cast<const scalar_t*>(
      static_cast<uintptr_t>(item[kGradientPointer]));
  __shared__ float tile[64][65];
  constexpr int local_rows = Analyze ? 16 : 32;
  constexpr int row_parts = 64 / local_rows;
  constexpr int wave_count = Analyze ? 4 : 8;
  int linear_thread = threadIdx.y * 16 + threadIdx.x;
  int local_maximum = 0;
  int local_count = 0;
  int local_nonfinite = 0;
  int local_invalid = 0;
  float local_total = 0.0f;
  for (int row_part = 0; row_part < row_parts; ++row_part) {
    int local_row = threadIdx.y + local_rows * row_part;
    int64_t row = tile_row * 64 + local_row;
    for (int column_part = 0; column_part < 4; ++column_part) {
      int local_column = threadIdx.x + 16 * column_part;
      int64_t column = tile_column * 64 + local_column;
      float square = 0.0f;
      if (row < rows && column < columns) {
        float value = static_cast<float>(gradient[row * columns + column]);
        if constexpr (Analyze) {
          if (!isfinite(value)) {
            local_nonfinite = 1;
          } else if (value != 0.0f) {
            ++local_count;
            local_maximum = max(local_maximum, positive_float_bits(fabsf(value)));
            square = value * value;
            if (!isfinite(square) || square == 0.0f) {
              local_invalid = 1;
              square = 0.0f;
            }
          }
        } else {
          square = value * value;
        }
      }
      tile[local_row][local_column] = square;
      local_total += square;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int row_part = 0; row_part < row_parts; ++row_part) {
      int local_row = threadIdx.y + local_rows * row_part;
      int64_t row = tile_row * 64 + local_row;
      if (row < rows) {
        float sum = 0.0f;
        for (int x = 0; x < 64; ++x) sum += tile[local_row][x];
        atomicAdd(row_energy + item[kRowOffset] + row, sum);
      }
    }
  }
  if (threadIdx.y == 0) {
    for (int column_part = 0; column_part < 4; ++column_part) {
      int local_column = threadIdx.x + 16 * column_part;
      int64_t column = tile_column * 64 + local_column;
      if (column < columns) {
        float sum = 0.0f;
        for (int y = 0; y < 64; ++y) sum += tile[y][local_column];
        atomicAdd(column_energy + item[kColumnOffset] + column, sum);
      }
    }
  }
  int lane = linear_thread & 63;
  int wave = linear_thread >> 6;
  for (int stride = 32; stride > 0; stride /= 2) {
    local_total += __shfl_down(local_total, stride, 64);
    if constexpr (Analyze) {
      local_maximum = max(local_maximum, __shfl_down(local_maximum, stride, 64));
      local_count += __shfl_down(local_count, stride, 64);
      local_nonfinite |= __shfl_down(local_nonfinite, stride, 64);
      local_invalid |= __shfl_down(local_invalid, stride, 64);
    }
  }
  __shared__ float wave_totals[8];
  __shared__ int wave_results[4][4];
  if (lane == 0) wave_totals[wave] = local_total;
  if constexpr (Analyze) {
    if (lane == 0) {
      wave_results[wave][0] = local_maximum;
      wave_results[wave][1] = local_count;
      wave_results[wave][2] = local_nonfinite;
      wave_results[wave][3] = local_invalid;
    }
  }
  __syncthreads();
  if (linear_thread == 0) {
    float tile_total = wave_totals[0];
    for (int other = 1; other < wave_count; ++other) tile_total += wave_totals[other];
    atomicAdd(total_energy + tensor, tile_total);
    if constexpr (Analyze) {
      for (int other = 1; other < 4; ++other) {
        wave_results[0][0] = max(wave_results[0][0], wave_results[other][0]);
        wave_results[0][1] += wave_results[other][1];
        wave_results[0][2] |= wave_results[other][2];
        wave_results[0][3] |= wave_results[other][3];
      }
      for (int field = 0; field < 4; ++field) {
        analysis_partials[blockIdx.x * 4 + field] = wave_results[0][field];
      }
    }
  }
}

__global__ void foreach_analysis_finalize_kernel(
    const int64_t* metadata,
    const int* analysis_partials,
    int* status,
    int tensor_count) {
  int tensor = blockIdx.x;
  const int64_t* item = metadata + tensor * kMetaFields;
  int begin = static_cast<int>(item[kTileOffset]);
  int end = begin + static_cast<int>(item[kTileCount]);
  int local_maximum = 0;
  int local_count = 0;
  int local_nonfinite = 0;
  int local_invalid = 0;
  for (int tile = begin + threadIdx.x; tile < end; tile += blockDim.x) {
    local_maximum = max(local_maximum, analysis_partials[tile * 4]);
    local_count += analysis_partials[tile * 4 + 1];
    local_nonfinite |= analysis_partials[tile * 4 + 2];
    local_invalid |= analysis_partials[tile * 4 + 3];
  }
  __shared__ int maxima[kThreads];
  __shared__ int counts[kThreads];
  __shared__ int nonfinite[kThreads];
  __shared__ int invalid[kThreads];
  maxima[threadIdx.x] = local_maximum;
  counts[threadIdx.x] = local_count;
  nonfinite[threadIdx.x] = local_nonfinite;
  invalid[threadIdx.x] = local_invalid;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
      counts[threadIdx.x] += counts[threadIdx.x + stride];
      nonfinite[threadIdx.x] |= nonfinite[threadIdx.x + stride];
      invalid[threadIdx.x] |= invalid[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    status[tensor * 4] = maxima[0];
    status[tensor * 4 + 1] = counts[0];
    status[tensor * 4 + 2] = nonfinite[0];
    status[tensor * 4 + 3] = invalid[0];
  }
}

template <typename scalar_t, bool Fast>
__global__ void foreach_raw_norm_kernel(
    const int64_t* metadata,
    int tensor_count,
    int* status,
    const float* inverse_maximum,
    const float* total_energy,
    const float* row_energy,
    const float* column_energy,
    float* norm_square) {
  int tensor = locate_metadata(metadata, tensor_count, blockIdx.x, kTileOffset);
  const int64_t* item = metadata + tensor * kMetaFields;
  if constexpr (!Fast) {
    if (status[tensor * 4 + 2] || status[tensor * 4 + 3] ||
        status[tensor * 4 + 1] <= 1) return;
  }
  int columns = static_cast<int>(item[kColumns]);
  int rows = static_cast<int>(item[kRows]);
  int local_tile = static_cast<int>(blockIdx.x - item[kTileOffset]);
  int tile_columns = (columns + 63) / 64;
  int tile_row = local_tile / tile_columns;
  int tile_column = local_tile % tile_columns;
  const scalar_t* gradient = reinterpret_cast<const scalar_t*>(
      static_cast<uintptr_t>(item[kGradientPointer]));
  float scale = inverse_maximum[tensor];
  float local_sum = 0.0f;
  int local_invalid = 0;
  int linear_thread = threadIdx.x;
  int thread_row = linear_thread / 16;
  int thread_column = linear_thread % 16;
  for (int row_part = 0; row_part < 2; ++row_part) {
    int row = tile_row * 64 + thread_row + 32 * row_part;
    if (row >= rows) continue;
    for (int column_part = 0; column_part < 4; ++column_part) {
      int column = tile_column * 64 + thread_column + 16 * column_part;
      if (column >= columns) continue;
      int index = row * columns + column;
      float value = static_cast<float>(gradient[index]);
      if constexpr (!Fast) {
        if (value == 0.0f) continue;
      }
      float denominator = sqrtf(row_energy[item[kRowOffset] + row] / static_cast<float>(columns)) +
          sqrtf(column_energy[item[kColumnOffset] + column] / static_cast<float>(rows));
      float raw = __fdividef(value * scale, denominator);
      float square = raw * raw;
      if constexpr (Fast) {
        local_sum += square;
      } else {
        if (!(denominator > 0.0f) || !isfinite(denominator) ||
            !isfinite(square) || square == 0.0f) {
          local_invalid = 1;
        } else {
          local_sum += square;
        }
      }
    }
  }
  for (int stride = 32; stride > 0; stride /= 2) {
    local_sum += __shfl_down(local_sum, stride, 64);
    local_invalid |= __shfl_down(local_invalid, stride, 64);
  }
  int lane = threadIdx.x & 63;
  int wave = threadIdx.x >> 6;
  __shared__ float wave_sums[8];
  __shared__ int wave_invalid[8];
  if (lane == 0) {
    wave_sums[wave] = local_sum;
    wave_invalid[wave] = local_invalid;
  }
  __syncthreads();
  float block_sum = threadIdx.x < 8 ? wave_sums[threadIdx.x] : 0.0f;
  int block_invalid = threadIdx.x < 8 ? wave_invalid[threadIdx.x] : 0;
  for (int stride = 32; stride > 0; stride /= 2) {
    block_sum += __shfl_down(block_sum, stride, 64);
    block_invalid |= __shfl_down(block_invalid, stride, 64);
  }
  if (threadIdx.x == 0) {
    atomicAdd(norm_square + tensor, block_sum);
    if constexpr (!Fast) atomicOr(status + tensor * 4 + 3, block_invalid);
  }
}

template <typename scalar_t, bool Fast>
__global__ void foreach_output_kernel(
    const int64_t* metadata,
    int tensor_count,
    int* status,
    const float* inverse_maximum,
    const float* total_energy,
    const float* row_energy,
    const float* column_energy,
    float learning_rate) {
  int tensor = locate_metadata(metadata, tensor_count, blockIdx.x, kTileOffset);
  const int64_t* item = metadata + tensor * kMetaFields;
  int columns = static_cast<int>(item[kColumns]);
  int rows = static_cast<int>(item[kRows]);
  int local_tile = static_cast<int>(blockIdx.x - item[kTileOffset]);
  int tile_columns = (columns + 63) / 64;
  int tile_row = local_tile / tile_columns;
  int tile_column = local_tile % tile_columns;
  scalar_t* parameter = reinterpret_cast<scalar_t*>(
      static_cast<uintptr_t>(item[kParameterPointer]));
  const scalar_t* gradient = reinterpret_cast<const scalar_t*>(
      static_cast<uintptr_t>(item[kGradientPointer]));
  float scale = inverse_maximum[tensor];
  int linear_thread = threadIdx.x;
  int thread_row = linear_thread / 32;
  int thread_column = linear_thread % 32;
  for (int row_part = 0; row_part < 4; ++row_part) {
    int row = tile_row * 64 + thread_row + 16 * row_part;
    if (row >= rows) continue;
    for (int column_part = 0; column_part < 2; ++column_part) {
      int column = tile_column * 64 + thread_column + 32 * column_part;
      if (column >= columns) continue;
      int index = row * columns + column;
      float value = static_cast<float>(gradient[index]);
      float direction = 0.0f;
      if constexpr (Fast) {
        float denominator = sqrtf(row_energy[item[kRowOffset] + row] / static_cast<float>(columns)) +
            sqrtf(column_energy[item[kColumnOffset] + column] / static_cast<float>(rows));
        direction = __fdividef(value * scale, denominator);
      } else {
        if (!status[tensor * 4 + 2] && !status[tensor * 4 + 3]) {
          if (status[tensor * 4 + 1] == 1 && value != 0.0f) {
            float radius = sqrtf(static_cast<float>(max(item[kRows], item[kColumns])));
            direction = copysignf(radius, value);
          } else if (status[tensor * 4 + 1] > 1 && value != 0.0f) {
            float denominator = sqrtf(row_energy[item[kRowOffset] + row] / static_cast<float>(columns)) +
                sqrtf(column_energy[item[kColumnOffset] + column] / static_cast<float>(rows));
            direction = __fdividef(value * scale, denominator);
          }
        }
      }
      if constexpr (Fast) {
        float current = static_cast<float>(parameter[index]);
        parameter[index] = static_cast<scalar_t>(current - learning_rate * direction);
      } else if (!status[tensor * 4 + 2] && !status[tensor * 4 + 3]) {
        float current = static_cast<float>(parameter[index]);
        parameter[index] = static_cast<scalar_t>(current - learning_rate * direction);
      }
    }
  }
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
  if (blockIdx.x == 0) {
    for (int64_t index = threadIdx.x; index < rows; index += blockDim.x) {
      outside_rows[index] = sqrtf(row_energy[index] / static_cast<float>(columns));
    }
  } else {
    for (int64_t index = threadIdx.x; index < columns; index += blockDim.x) {
      outside_columns[index] = sqrtf(column_energy[index] / static_cast<float>(rows));
    }
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
    int64_t maximum_dimension,
    const int* status,
    const float* outside_rows,
    const float* outside_columns,
    const int* minimum_bits,
    const float* norm_square,
    float learning_rate) {
  float maximum = bits_float(status[0]);
  float minimum = bits_float(minimum_bits[0]);
  float radius = sqrtf(static_cast<float>(maximum_dimension));
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
        std::max(rows, columns),
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
        std::max(rows, columns),
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
        std::max(rows, columns),
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
        std::max(rows, columns),
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

at::Tensor cauchylift_foreach_step_hip(
    at::TensorList parameters,
    at::TensorList gradients,
    double learning_rate,
    bool prevalidated_dense) {
  TORCH_CHECK(!parameters.empty(), "CauchyLift foreach requires parameters");
  TORCH_CHECK(parameters.size() == gradients.size(), "parameter and gradient list sizes differ");
  const at::Tensor& first = gradients[0];
  check_input(first);
  c10::cuda::CUDAGuard guard(first.device());
  int tensor_count = static_cast<int>(parameters.size());
  int64_t total_rows = 0;
  int64_t total_columns = 0;
  int64_t total_tiles = 0;
  auto metadata_cpu = at::empty(
      {tensor_count, kMetaFields},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* metadata_host = metadata_cpu.data_ptr<int64_t>();
  for (int tensor = 0; tensor < tensor_count; ++tensor) {
    const at::Tensor& parameter = parameters[tensor];
    const at::Tensor& gradient = gradients[tensor];
    check_input(parameter);
    check_input(gradient);
    TORCH_CHECK(parameter.sizes() == gradient.sizes(), "parameter and gradient shapes differ");
    TORCH_CHECK(parameter.scalar_type() == gradient.scalar_type(), "parameter and gradient dtypes differ");
    TORCH_CHECK(parameter.scalar_type() == first.scalar_type(), "foreach tensors must share a dtype");
    TORCH_CHECK(parameter.device() == first.device(), "foreach tensors must share a device");
    auto [rows, columns] = matrix_shape(gradient);
    int64_t tiles = ((rows + 63) / 64) * ((columns + 63) / 64);
    int64_t* item = metadata_host + tensor * kMetaFields;
    item[kParameterPointer] = reinterpret_cast<int64_t>(parameter.data_ptr());
    item[kGradientPointer] = reinterpret_cast<int64_t>(gradient.data_ptr());
    item[kRows] = rows;
    item[kColumns] = columns;
    item[kRowOffset] = total_rows;
    item[kColumnOffset] = total_columns;
    item[kTileOffset] = total_tiles;
    item[kTileCount] = tiles;
    total_rows += rows;
    total_columns += columns;
    total_tiles += tiles;
  }
  auto metadata = metadata_cpu.to(first.device(), at::kLong, true, true);
  auto float_options = first.options().dtype(at::kFloat);
  auto int_options = first.options().dtype(at::kInt);
  auto status = at::empty({tensor_count, 4}, int_options);
  auto norm_square = at::empty({tensor_count}, float_options);
  auto inverse_maximum = at::empty({tensor_count}, float_options);
  auto total_energy = at::empty({tensor_count}, float_options);
  auto row_energy = at::empty({total_rows}, float_options);
  auto column_energy = at::empty({total_columns}, float_options);
  at::Tensor analysis_partials;
  if (!prevalidated_dense) {
    analysis_partials = at::empty({total_tiles, 4}, int_options);
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int initialize_blocks = static_cast<int>(std::min<int64_t>(
      1024,
      (std::max(
           total_rows,
           std::max(total_columns, static_cast<int64_t>(tensor_count))) +
       kThreads - 1) /
          kThreads));
  foreach_initialize_kernel<<<initialize_blocks, kThreads, 0, stream>>>(
      status.data_ptr<int>(),
      norm_square.data_ptr<float>(),
      inverse_maximum.data_ptr<float>(),
      total_energy.data_ptr<float>(),
      row_energy.data_ptr<float>(),
      total_rows,
      column_energy.data_ptr<float>(),
      total_columns,
      tensor_count,
      prevalidated_dense);
  dim3 fast_marginal_threads(16, 32);
  dim3 safe_marginal_threads(16, 16);
  if (first.scalar_type() == at::kFloat) {
    if (prevalidated_dense) {
      foreach_marginal_energy_kernel<float, false><<<total_tiles, fast_marginal_threads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, nullptr,
          total_energy.data_ptr<float>(), row_energy.data_ptr<float>(),
          column_energy.data_ptr<float>());
    } else {
      foreach_marginal_energy_kernel<float, true><<<total_tiles, safe_marginal_threads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, analysis_partials.data_ptr<int>(),
          total_energy.data_ptr<float>(), row_energy.data_ptr<float>(),
          column_energy.data_ptr<float>());
    }
  } else {
    if (prevalidated_dense) {
      foreach_marginal_energy_kernel<at::BFloat16, false><<<total_tiles, fast_marginal_threads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, nullptr,
          total_energy.data_ptr<float>(), row_energy.data_ptr<float>(),
          column_energy.data_ptr<float>());
    } else {
      foreach_marginal_energy_kernel<at::BFloat16, true><<<total_tiles, safe_marginal_threads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, analysis_partials.data_ptr<int>(),
          total_energy.data_ptr<float>(), row_energy.data_ptr<float>(),
          column_energy.data_ptr<float>());
    }
  }
  if (!prevalidated_dense) {
    foreach_analysis_finalize_kernel<<<tensor_count, kThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(), analysis_partials.data_ptr<int>(),
        status.data_ptr<int>(), tensor_count);
    foreach_inverse_maximum_kernel<<<1, kThreads, 0, stream>>>(
        status.data_ptr<int>(), inverse_maximum.data_ptr<float>(), tensor_count);
  }
  if (first.scalar_type() == at::kFloat) {
    if (prevalidated_dense) {
      foreach_raw_norm_kernel<float, true><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          norm_square.data_ptr<float>());
    } else {
      foreach_raw_norm_kernel<float, false><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          norm_square.data_ptr<float>());
    }
    foreach_finalize_scale_kernel<<<1, kThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(), status.data_ptr<int>(),
        inverse_maximum.data_ptr<float>(), norm_square.data_ptr<float>(), tensor_count);
    if (prevalidated_dense) {
      foreach_output_kernel<float, true><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          static_cast<float>(learning_rate));
    } else {
      foreach_output_kernel<float, false><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          static_cast<float>(learning_rate));
    }
  } else {
    if (prevalidated_dense) {
      foreach_raw_norm_kernel<at::BFloat16, true><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          norm_square.data_ptr<float>());
    } else {
      foreach_raw_norm_kernel<at::BFloat16, false><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          norm_square.data_ptr<float>());
    }
    foreach_finalize_scale_kernel<<<1, kThreads, 0, stream>>>(
        metadata.data_ptr<int64_t>(), status.data_ptr<int>(),
        inverse_maximum.data_ptr<float>(), norm_square.data_ptr<float>(), tensor_count);
    if (prevalidated_dense) {
      foreach_output_kernel<at::BFloat16, true><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          static_cast<float>(learning_rate));
    } else {
      foreach_output_kernel<at::BFloat16, false><<<total_tiles, kElementThreads, 0, stream>>>(
          metadata.data_ptr<int64_t>(), tensor_count, status.data_ptr<int>(),
          inverse_maximum.data_ptr<float>(), total_energy.data_ptr<float>(),
          row_energy.data_ptr<float>(), column_energy.data_ptr<float>(),
          static_cast<float>(learning_rate));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return status;
}

TORCH_LIBRARY(cauchylift_native, module) {
  module.def("direction(Tensor gradient) -> (Tensor, Tensor)");
  module.def("step_(Tensor(a!) parameter, Tensor gradient, float learning_rate) -> Tensor");
  module.def("foreach_step_(Tensor(a!)[] parameters, Tensor[] gradients, float learning_rate, bool prevalidated_dense) -> Tensor");
}

TORCH_LIBRARY_IMPL(cauchylift_native, CUDA, module) {
  module.impl("direction", &cauchylift_direction_hip);
  module.impl("step_", &cauchylift_step_hip);
  module.impl("foreach_step_", &cauchylift_foreach_step_hip);
}
