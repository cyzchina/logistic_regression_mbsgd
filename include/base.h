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
  uint32_t *index;
  size_t start;
  size_t end;
  float y3;
  float mu;
  float yita;
  float *weights;
  float *total_l1;
  float *old_pd;
  float *v;
  float *z;
  float **batch_data;
  const TRAIN_ARG *parg_train;
} TASK_ARG;

