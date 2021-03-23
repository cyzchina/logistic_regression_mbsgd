#include <sched.h>
#include <pthread.h>

#include "base.h"
#include "random.h"
#include "task.h"
#include "lr.h"

void
train(const TRAIN_ARG *parg) {
  size_t i, j;

//  printf("*************** 1\n");
//#ifdef _PYTHON_MBSGD
//  for (i = 0; i < 4; ++i) {
//    for (j = 0; j < parg->feature_size; ++j) {
//      printf("%.17g ", parg->data[i + j * parg->data_size]);
//    }
//    printf("\n");
//  }
//#else
//  for (i = 0; i < 4; ++i) {
//    for (j = 0; j < parg->feature_size; ++j) {
//      printf("%.17g ", parg->data[i][j]);
//    }
//    printf("\n");
//  }
//#endif
//  printf("*************** 2\n");

  double *weights = (double*)calloc(parg->feature_size, sizeof(double));

  size_t weights_size = sizeof(double) * parg->feature_size;
  uint32_t *randoms = (uint32_t*)calloc(parg->feature_size > parg->data_size? parg->feature_size:parg->data_size, sizeof(uint32_t));
  if (parg->randw) {
    create_randoms(randoms, parg->feature_size);

    for (i = 0; i < parg->feature_size; ++i) {
      weights[i] = -1.0 + 2.0 * ((double)randoms[i] / UINT32_MAX); 
    }
  }

#ifndef _PYTHON_MBSGD
  printf("\n# stochastic gradient descent\n");
#endif

  bool sprint = false;
  double norm = 1.0;
  double min_norm;
  double old_norm = 1.0;
//#ifndef _PYTHON_MBSGD
  double l1n = 0;
//#endif
  double mu = 0;
  //double gama = 0.9;
  double y1 = pow(59, -1.0 / parg->data_size); 
  double y2 = 1.0;
  double y3;
  double yita;
  double c, d;
  size_t n = 0;
  size_t sprint_maxit = parg->maxit + 30;

  double *old_weights = (double*)calloc(parg->feature_size, sizeof(double));
  double *total_l1 = (double*)calloc(parg->feature_size, sizeof(double));

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

  cpu_set_t mask;
  pthread_t *threads = (pthread_t*)calloc(parg->cpus, sizeof(pthread_t));
  pthread_attr_t *attrs = (pthread_attr_t*)calloc(parg->cpus, sizeof(pthread_attr_t));
  TASK_ARG *task_args = (TASK_ARG*)calloc(parg->cpus, sizeof(TASK_ARG));
  size_t pthread_batch_size = parg->data_size / parg->cpus;
  size_t pthread_heavy = parg->data_size % parg->cpus;
  size_t pthread_count = parg->cpus, pthread_idx = 0;

  for (i = 0; i < parg->cpus; ++i) {
    task_args[i].start = pthread_idx;
    if (i < pthread_heavy) {
      pthread_idx += pthread_batch_size + 1;
    }
    else if (pthread_batch_size > 0) {
      pthread_idx += pthread_batch_size;
    }
    else {
      pthread_count = i;
      break;
    }
    task_args[i].end = pthread_idx;

    CPU_ZERO(&mask);
    CPU_SET(i, &mask);
    pthread_attr_init(&attrs[i]);
    pthread_attr_setaffinity_np(&attrs[i], sizeof(mask), &mask);

    task_args[i].parg_train = parg;
    task_args[i].index = index;
    task_args[i].weights = (double*)calloc(parg->feature_size, sizeof(double));
    task_args[i].total_l1 = (double*)calloc(parg->feature_size, sizeof(double));
    task_args[i].old_pd = (double*)calloc(parg->feature_size, sizeof(double));
    task_args[i].v = (double*)calloc(parg->feature_size, sizeof(double));
    task_args[i].z = (double*)calloc(orig_batch, sizeof(double));
    task_args[i].batch_data = (double**)calloc(orig_batch, sizeof(double*));
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

    //for (i = 0; i < parg->feature_size; ++i) {
    //  printf("%f ", weights[i]);
    //}
    //printf("\n");

    //for (i = 0; i < parg->feature_size; ++i) {
    //  printf("%f ", old_weights[i]);
    //}
    //printf("\n");

    norm = vecnorm(weights, old_weights, parg->feature_size);

//#ifndef _PYTHON_MBSGD
    if (n && n % 100 == 0) {       
      l1n = l1norm(weights, parg->feature_size);
      printf("# convergence: %1.4f l1-norm: %1.4e iterations: %lu batch: %d\n", norm, l1n, n, batch);     
    }
//#endif

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
      //task_args[0].task_batch = 1;
      //task_args[0].mu = mu;
      //task_args[0].start = 0;
      //task_args[0].end = parg->data_size;
      //task_args[0].mu = 0;
      //memcpy(task_args[0].weights, weights, weights_size);
      //memcpy(task_args[0].total_l1, total_l1, weights_size);
      //memset(task_args[0].total_l1, 0, weights_size);
      //y2 = 1.0;
    }               
  }

//#ifndef _PYTHON_MBSGD
  l1n = l1norm(weights, parg->feature_size);
  printf("# convergence: %1.4f l1-norm: %1.4e iterations: %lu batch: %d\n", norm, l1n, n, batch);     
//#endif

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
