#include "base.h"
#include <math.h>
#include <cuda_runtime.h>

typedef float (*PF_HANDLE)(float, const void*);

//static float
//sigmoid(float x) {
//  static float overflow = 20.0;
//  if (x > overflow) x = overflow;
//  if (x < -overflow) x = -overflow;
//
//  return 1.0/(1.0 + exp(-x));
//}

static __device__ float
cu_sigmoid(float x) {
  static float overflow = 20.0;
  if (x > overflow) x = overflow;
  if (x < -overflow) x = -overflow;

  return 1.0 / (1.0 + exp(-x));
}

static __device__ float
cu_z(float val, const void *params) {
  return cu_sigmoid(val) - ((float*)params)[blockIdx.x];
}

__device__ PF_HANDLE pf_handles[] = {NULL, cu_z};

static __device__ void
reduce(float val, unsigned short pf_idx, const void *params, float *out) {
  extern __shared__ float sums[];
  unsigned int s;
  if (blockDim.x > 32) {
    s = 16;
  }
  else {
    s = blockDim.x >> 1; 
  }
  for (; s > 0; s >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, s);
  }

  if (0 == (threadIdx.x & 0x1f)) {
    unsigned int offset = blockIdx.x * WARP_COUNT;
    sums[offset + (threadIdx.x >> 5)] = val;

    __syncthreads();
    if (0 == threadIdx.x) {
      val = 0;
      for (s = 0; s < WARP_COUNT; ++s) {
        val += sums[offset + s];
      }

      if (1 == gridDim.y && pf_idx > 0) {
        val = pf_handles[pf_idx](val, params);
      }

      out[blockIdx.x * gridDim.y + blockIdx.y] = val;
    }
  }
}

static __global__ void
reduce_weighted_sum(const float *in, const float *weights, unsigned short pf_idx, const void *params, float *out, unsigned int count) {
  float val = 0;
  unsigned int idx = (blockIdx.y << 1) * blockDim.x + threadIdx.x;
  if (idx < count) {
    unsigned int offset = blockIdx.x * count;
    val = in[offset + idx] * weights[idx];
    idx += blockDim.x;
    if (idx < count) {
      val += in[offset + idx] * weights[idx];
    }
  }

  reduce(val, pf_idx, params, out);
}

static __global__ void
reduce_sum(const float *in, unsigned short pf_idx, const void *params, float *out, unsigned int count) {
  float val = 0;
  unsigned int idx = (blockIdx.y << 1) * blockDim.x + threadIdx.x;
  if (idx < count) {
    unsigned int offset = blockIdx.x * count;
    val = in[offset + idx];
    idx += blockDim.x;
    if (idx < count) {
      val += in[offset + idx];
    }
  }

  reduce(val, pf_idx, params, out);
}

static __global__ void
cu_delta_weights(const float *in, const float *data, float *delta_weights, unsigned int task_batch, unsigned int feature_size, unsigned int data_size, float gama) {
  unsigned int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (feature_idx >= feature_size) {
    return;
  }

  float val = 0;
  unsigned int data_idx = blockIdx.x * task_batch;
  unsigned int i;
  for (i = 0; i < task_batch; ++i) {
    if (data_idx >= data_size) {
      break;
    }
    val += in[data_idx] * data[data_idx * feature_size + feature_idx];
    ++data_idx;
  } 
  val /= i;

  atomicAdd(&delta_weights[feature_idx], gama * val);
}

static __global__ void
cu_adjust_weights(float *weights, float *delta_weights, unsigned int batch_count, unsigned int feature_size, float *norm) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= feature_size) {
    return;
  }
  float val = delta_weights[idx] / batch_count;
  weights[idx] -= val;
  atomicAdd(norm, val * val);
}

extern "C" {
float
gpu_task(TASK_ARG *parg) {

  dim3 block(BLOCK_SIZE), grid;
  unsigned short pf_idx = 1;

  unsigned int rec_count;
  for (size_t i = 0; i < parg->parg_train->data_size; i += rec_count) {
    rec_count = parg->parg_train->data_size - i;
    if (rec_count > parg->max_rec_count) {
        rec_count = parg->max_rec_count;
    }

    float *d_in = &parg->d_data[i * parg->parg_train->feature_size]; 
    float *d_labels = &parg->d_labels[i];
    unsigned int count = parg->parg_train->feature_size;
    size_t shared_size = sizeof(float) * WARP_COUNT * rec_count;
    grid.x = rec_count;

    unsigned int block_count = (parg->block_count + 1) >> 1;
    if (1 == block_count) {
      grid.y = block_count;
      reduce_weighted_sum<<<grid, block, shared_size>>>(d_in, parg->d_weights, pf_idx, d_labels, &parg->d_out[i], count);
      continue;
    }

    bool b_first = true;
    float *d_tmp = parg->d_tmp;
    while (count > 1) {
      grid.y = block_count;
      if (1 == block_count) {
        reduce_sum<<<grid, block, shared_size>>>(d_in, pf_idx, d_labels, &parg->d_out[i], count);
        count = 1;
      }
      else {
        if (b_first) {
          reduce_weighted_sum<<<grid, block, shared_size>>>(d_in, parg->d_weights, pf_idx, d_labels, d_tmp, count);
          b_first = false;
        }
        else {
          reduce_sum<<<grid, block, shared_size>>>(d_in, pf_idx, d_labels, d_tmp, count);
        }

        count = block_count;
        block_count = (block_count + BLOCK_SIZE - 1) / BLOCK_SIZE; 
        block_count = (block_count + 1) >> 1;

        d_in = d_tmp;
        if (parg->d_tmp == d_in) {
          d_tmp = &d_in[count * rec_count];
        }
        else {
          d_tmp = parg->d_tmp;
        }
      }
    }
  }

  //float *out = (float*)calloc(parg->parg_train->data_size, sizeof(float));
  //float *weights = (float*)calloc(parg->parg_train->feature_size, sizeof(float));
  //float *data = (float*)calloc(parg->parg_train->feature_size * parg->parg_train->data_size, sizeof(float));
  //float *labels = (float*)calloc(rec_count, sizeof(float));
  //cudaMemcpy(out, parg->d_out, sizeof(float) * parg->parg_train->data_size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(weights, parg->d_weights, sizeof(float) * parg->parg_train->feature_size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(data, parg->d_data, sizeof(float) * parg->parg_train->feature_size * parg->parg_train->data_size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(labels, parg->d_labels, sizeof(float) * rec_count, cudaMemcpyDeviceToHost);
  //for (int j = 0; j < parg->parg_train->data_size; ++j) {
  //  float sum = 0;
  //  for (int k = 0; k < parg->parg_train->feature_size; ++k) {
  //    sum += data[j * parg->parg_train->feature_size + k] * weights[k];
  //  }
  //  sum = sigmoid(sum) - labels[j];
  //  if (fabs(sum - out[j]) > 1e-5) {
  //    printf("%d: %f %f\n", j, sum, out[j]);
  //  }
  //}
  //free(out);
  //free(weights);
  //free(data);
  //free(labels);

  unsigned int batch_count = (parg->parg_train->data_size + parg->task_batch - 1) / parg->task_batch;
  grid.x = batch_count;
  grid.y = parg->block_count;
  cudaMemset(parg->d_delta_weights, 0, sizeof(float) * parg->parg_train->feature_size);
  cu_delta_weights<<<grid, block>>>(parg->d_out, parg->d_data, parg->d_delta_weights, parg->task_batch, parg->parg_train->feature_size, parg->parg_train->data_size, parg->parg_train->gama);

  grid.x = parg->block_count;
  grid.y = 1;
  cudaMemset(parg->d_norm, 0, sizeof(float));
  cu_adjust_weights<<<grid, block>>>(parg->d_weights, parg->d_delta_weights, batch_count, parg->parg_train->feature_size, parg->d_norm);

  float norm;
  cudaMemcpy(&norm, parg->d_norm, sizeof(float), cudaMemcpyDeviceToHost);
  return sqrt(norm);
}
}
