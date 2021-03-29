#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

typedef struct {
  int shuf;
  int randw;
  unsigned int maxit;
  unsigned short cpus;
  #ifdef _CUDA
  float alpha;
  float gama;
  float l1;
  float eps;
  float *labels;
  float **data;
  float *sprint_weights;
  #else
  double alpha;
  double gama;
  double l1;
  double eps;
  double *labels;
  double **data;
  double *sprint_weights;
  #endif
  size_t data_size;
  size_t feature_size;
} TRAIN_ARG;

typedef struct {
  uint32_t task_batch;
  uint32_t *index;
  size_t start;
  size_t end;
  #ifdef _CUDA
  float y3;
  float mu;
  float yita;
  float *weights;
  float *total_l1;
  float *old_pd;
  float *v;
  float *z;
  float **batch_data;
  #else
  double y3;
  double mu;
  double yita;
  double *weights;
  double *total_l1;
  double *old_pd;
  double *v;
  double *z;
  double **batch_data;
  #endif
  const TRAIN_ARG *parg_train;
} TASK_ARG;

