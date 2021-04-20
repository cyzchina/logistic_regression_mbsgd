#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

#ifdef _CUDA
static const uint32_t BLOCK_SIZE = 32;
static const uint32_t WARP_COUNT = (BLOCK_SIZE + 31 ) >> 5;
//#include <cuda_runtime.h>
#endif


typedef struct {
  int shuf;
  int randw;
  unsigned int maxit;
  #ifndef _CUDA
  unsigned short cpus;
  #endif
  float alpha;
  float gama;
  float l1;
  float eps;
  float *labels;
  float **data;
  float *sprint_weights;
  size_t data_size;
  size_t feature_size;
} TRAIN_ARG;

typedef struct {
  uint32_t task_batch;
  float mu;
  #ifdef _CUDA
  uint32_t block_count;
  size_t max_rec_count;
  float *d_labels;
  float *d_data;
  float *d_weights;
  float *d_delta_weights;
  float *d_sprint_weights;
  float *d_out;
  float *d_norm;
  float *d_total_l1;
  #else
  uint32_t *index;
  float y3;
  float yita;
  size_t start;
  size_t end;
  float *weights;
  float *z;
  float *total_l1;
  float *old_pd;
  float *v;
  float **batch_data;
  #endif
  const TRAIN_ARG *parg_train;
} TASK_ARG;

