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
  double alpha;
  double gama;
  double l1;
  unsigned int maxit;
  int shuf;
  unsigned short cpus;
  double eps;
  int randw;
  double *labels;
#ifdef _PYTHON_MBSGD
  double *data;
#else
  double **data;
#endif
  size_t data_size;
  size_t feature_size;
  double *sprint_weights;
} TRAIN_ARG;

typedef struct {
  uint32_t task_batch;
  size_t start;
  size_t end;
  double y3;
  double mu;
  double yita;
  uint32_t *index;
  double *weights;
  double *total_l1;
  double *old_pd;
  double *v;
  double *z;
  double **batch_data;
  const TRAIN_ARG *parg_train;
} TASK_ARG;

