#include <sched.h>
#include <pthread.h>

#include "base.h"
#include "curandom.h"
#include "cutask.h"

//static const size_t MIN_BATCH_SIZE = 200;

extern "C" {
void
gpu_train(const TRAIN_ARG *parg) {
  #ifndef _PYTHON_MBSGD
  printf("\n# stochastic gradient descent\n");
  #endif

  cudaSetDevice(0);

  size_t i;

  size_t weights_size = sizeof(float) * parg->feature_size;

  float *d_weights;
  cudaMalloc(&d_weights, weights_size);

  float *weights = NULL; 
  uint32_t *randoms = NULL;
  if (parg->randw) {
    randoms = (uint32_t*)calloc(parg->feature_size > parg->data_size ? parg->feature_size:parg->data_size, sizeof(uint32_t));
    gpu_create_randoms(randoms, parg->feature_size);

    weights = (float*)calloc(parg->feature_size, sizeof(float));
    for (i = 0; i < parg->feature_size; ++i) {
      weights[i] = -1.0 + 2.0 * ((float)randoms[i] / UINT32_MAX); 
    }
    cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice);
  }
  else {
    cudaMemset(d_weights, 0, weights_size);
  }

  bool sprint = false;
  size_t n = 0;
  size_t sprint_maxit = parg->maxit + 30;

  float norm = 1.0;
  float min_norm;
  float old_norm = 1.0;

  float mu = 0;
  //float y1 = pow(59, -1.0 / parg->data_size); 
  //float y2 = 1.0;
  //float y3;
  //float yita;
  float c, d;

  //uint32_t *index = (uint32_t*)calloc(parg->data_size, sizeof(uint32_t));

  uint32_t batch;
  uint32_t orig_batch = (uint32_t)log(parg->data_size / 10);
  if (orig_batch <= 0) {
    orig_batch = 1;
  }
  batch = orig_batch;

  //for (i = 1; i < parg->data_size; ++i) {
  //  index[i] = i;
  //}

  float *d_labels;
  cudaMalloc(&d_labels, sizeof(float) * parg->data_size);
  cudaMemcpy(d_labels, parg->labels, sizeof(float) * parg->data_size, cudaMemcpyHostToDevice);

  float *d_data;
  cudaMalloc(&d_data, weights_size * parg->data_size);
  for (i = 0; i < parg->data_size; ++i) {
    cudaMemcpy(&d_data[i * parg->feature_size], parg->data[i], weights_size, cudaMemcpyHostToDevice);
  }


  float *d_delta_weights;
  cudaMalloc(&d_delta_weights, weights_size);

  float *d_norm;
  cudaMalloc(&d_norm, sizeof(float));

  float *d_sprint_weights = NULL;
  TASK_ARG task_arg = {
    .d_labels = d_labels,
    .d_data = d_data,
    .d_weights = d_weights,
    .d_delta_weights = d_delta_weights,
    .d_norm = d_norm,
    .parg_train = parg
  };

  uint32_t block_count = (parg->feature_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  task_arg.block_count = block_count;

  float *d_out;
  block_count = (block_count + 1) >> 1;
  cudaMalloc(&d_out, sizeof(float) * (block_count + ((block_count + 1) >> 1)) * parg->data_size);
  task_arg.d_out = d_out;

  while (norm > parg->eps) {
    //if (parg->shuf) {
    //  gpu_shuffle(index, randoms, parg->data_size);
    //}

    if (!sprint) {
      d = norm / parg->eps;
      if (d > 70) {
        if (batch < orig_batch) {
          ++batch;
        }
      }
      else if (d < 60) {
        c = norm / old_norm;
        if (c > 1.7) {
          if (batch < orig_batch) {
            ++batch;
          }
        }
        else if (c < 0.7) {
          if (batch > 1) {
            --batch;
          }
        }
      }
      else if (d < 10) {
        batch = 1;
      }
      old_norm = norm;
    }

    task_arg.task_batch = batch;
    task_arg.mu = mu;

    cudaMemset(d_delta_weights, 0, weights_size);
    norm = gpu_task(&task_arg);

    ++n;
    if (sprint) {
      if (norm < min_norm) {
        min_norm = norm;
        cudaMemcpy(d_sprint_weights, d_weights, weights_size, cudaMemcpyDeviceToDevice);
      }

      if(n > sprint_maxit) {
        break;
      }
    }
    else if (n > parg->maxit) {
      sprint = true;
      min_norm = norm;

      batch = 1;
      cudaMalloc(&d_sprint_weights, weights_size);
      cudaMemcpy(d_sprint_weights, d_weights, weights_size, cudaMemcpyDeviceToDevice);
    }               
  }

  if (sprint) {
    cudaMemcpy(parg->sprint_weights, d_sprint_weights, weights_size, cudaMemcpyDeviceToHost);
  }
  else {
    cudaMemcpy(parg->sprint_weights, d_weights, weights_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_labels);
  cudaFree(d_data);
  cudaFree(d_weights);
  cudaFree(d_delta_weights);
  cudaFree(d_sprint_weights);
  cudaFree(d_out);
  cudaFree(d_norm);

  free(randoms);
  free(weights);
}
}
