#include <sched.h>
#include <pthread.h>

#include "base.h"
#include "random.h"
#include "task.h"
#include "lr.h"

static const size_t MIN_BATCH_SIZE = 200;

void
train(const TRAIN_ARG *parg) {
  size_t i, j;

  float *weights = (float*)calloc(parg->feature_size, sizeof(float));
  size_t weights_size = sizeof(float) * parg->feature_size;

  uint32_t *randoms =  NULL;
  if (parg->randw || parg->shuf) {
    randoms = (uint32_t*)calloc(parg->feature_size > parg->data_size ? parg->feature_size:parg->data_size, sizeof(uint32_t));
    create_randoms(randoms, parg->feature_size);

    if (parg->randw) {
      for (i = 0; i < parg->feature_size; ++i) {
        weights[i] = -1.0 + 2.0 * ((float)randoms[i] / UINT32_MAX); 
      }
    }
  }

  #ifndef _PYTHON_MBSGD
  printf("\n# stochastic gradient descent\n");
  #endif

  bool sprint = false;
  size_t n = 0;
  size_t sprint_maxit = parg->maxit + 30;

  float norm = 1.0;
  float min_norm;
  float old_norm = 1.0;
  #ifndef _PYTHON_MBSGD
  float l1n = 0;
  #endif
  float mu = 0;
  float y1 = pow(59, -1.0 / parg->data_size); 
  float y2 = 1.0;
  float y3;
  float yita;
  float c, d;

  float *old_weights = (float*)calloc(parg->feature_size, sizeof(float));
  float *total_l1 = (float*)calloc(parg->feature_size, sizeof(float));

  uint32_t *index = (uint32_t*)calloc(parg->data_size, sizeof(uint32_t));

  uint32_t batch;
  uint32_t orig_batch = (uint32_t)(log(parg->data_size / 10));
  if (orig_batch <= 0) {
    orig_batch = 1;
  }
  batch = orig_batch;

  for (i = 1; i < parg->data_size; ++i) {
    index[i] = i;
  }

  size_t pthread_batch_size, pthread_heavy, pthread_count;
  size_t pthread_idx = 0;
  if (parg->data_size < MIN_BATCH_SIZE) {
    pthread_count = 1;
    pthread_batch_size = parg->data_size;
    pthread_heavy = 0;
  }
  else {
    pthread_count = parg->data_size / MIN_BATCH_SIZE;
    if (pthread_count >= parg->cpus) {
      pthread_count = parg->cpus;
    }
    pthread_batch_size = parg->data_size / pthread_count;
    pthread_heavy = parg->data_size % pthread_count;
  }

  cpu_set_t mask;
  pthread_t *threads = (pthread_t*)calloc(pthread_count, sizeof(pthread_t));
  pthread_attr_t *attrs = (pthread_attr_t*)calloc(pthread_count, sizeof(pthread_attr_t));
  TASK_ARG *task_args = (TASK_ARG*)calloc(pthread_count, sizeof(TASK_ARG));

  for (i = 0; i < pthread_count; ++i) {
    task_args[i].start = pthread_idx;
    pthread_idx += pthread_batch_size;
    if (i < pthread_heavy) {
      pthread_idx += 1;
    }
    task_args[i].end = pthread_idx;

    CPU_ZERO(&mask);
    CPU_SET(i, &mask);
    pthread_attr_init(&attrs[i]);
    pthread_attr_setaffinity_np(&attrs[i], sizeof(mask), &mask);

    task_args[i].parg_train = parg;
    task_args[i].index = index;
    task_args[i].weights = (float*)calloc(parg->feature_size, sizeof(float));
    task_args[i].total_l1 = (float*)calloc(parg->feature_size, sizeof(float));
    task_args[i].old_pd = (float*)calloc(parg->feature_size, sizeof(float));
    task_args[i].v = (float*)calloc(parg->feature_size, sizeof(float));
    task_args[i].z = (float*)calloc(orig_batch, sizeof(float));
    task_args[i].batch_data = (float**)calloc(orig_batch, sizeof(float*));
  }

  while (norm > parg->eps) {
    if (parg->shuf) {
      shuffle(index, randoms, parg->data_size);
    }

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

    memcpy(old_weights, weights, weights_size);

    y2 *= y1;
    yita = parg->alpha * y2;
    y3 = yita * parg->l1;

    for (i = 0; i < pthread_count; ++i) {
      task_args[i].task_batch = batch;
      task_args[i].y3 = y3;
      task_args[i].mu = mu;
      task_args[i].yita = yita;
      memcpy(task_args[i].weights, weights, weights_size);
      memcpy(task_args[i].total_l1, total_l1, weights_size);
      pthread_create(&threads[i], &attrs[i], task, &task_args[i]); 
    }

    memset(weights, 0, weights_size);
    memset(total_l1, 0, weights_size);
    mu = 0;
    for (i = 0; i < pthread_count; ++i) {
      pthread_join(threads[i], NULL);
      mu += task_args[i].mu / pthread_count; 
      for (j = 0; j < parg->feature_size; ++j) {
        total_l1[j] += task_args[i].total_l1[j] / pthread_count;
        weights[j] += task_args[i].weights[j] / pthread_count;
      }
    }

    norm = vecnorm(weights, old_weights, parg->feature_size);

    #ifndef _PYTHON_MBSGD
    if (n && n % 100 == 0) {       
      l1n = l1norm(weights, parg->feature_size);
      printf("# convergence: %1.4f l1-norm: %1.4e iterations: %lu batch: %d\n", norm, l1n, n, batch);     
    }
    #endif

    ++n;
    if (sprint) {
      if (norm < min_norm) {
        min_norm = norm;
        memcpy(parg->sprint_weights, task_args[0].weights, weights_size);
      }

      if(n > sprint_maxit) {
        break;
      }
    }
    else if (n > parg->maxit) {
      sprint = true;
      min_norm = norm;
      memcpy(parg->sprint_weights, weights, weights_size);

      batch = 1;
    }               
  }

#ifndef _PYTHON_MBSGD
  l1n = l1norm(weights, parg->feature_size);
  printf("# convergence: %1.4f l1-norm: %1.4e iterations: %lu batch: %d\n", norm, l1n, n, batch);     
#endif

  if (!sprint) {
    memcpy(parg->sprint_weights, weights, weights_size);
  }

  for (i = 0; i < pthread_count; ++i) {
    free(task_args[i].weights);
    free(task_args[i].total_l1);
    free(task_args[i].old_pd);
    free(task_args[i].v);
    free(task_args[i].z);
    free(task_args[i].batch_data);
    pthread_attr_destroy(&attrs[i]); 
  }

  free(threads);
  free(attrs);
  free(task_args);

  free(randoms);
  free(index);
  free(old_weights);
  free(weights);
  free(total_l1);
}
