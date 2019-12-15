#include "base.h"
#include "lr.h"

void*
task(void *param) {
  TASK_ARG *parg = (TASK_ARG*)param;

  uint32_t last_batch, cur_batch;
  size_t i, j, k;

  double predicted, a, b, pd;

  double **batch_data = (double**)calloc(parg->task_batch, sizeof(double*));
  double *z = (double*)calloc(parg->task_batch, sizeof(double));
  double *old_pd = (double*)calloc(parg->parg_train->feature_size, sizeof(double));
  double *v = (double*)calloc(parg->parg_train->feature_size, sizeof(double));

  i = parg->start;
  while (i < parg->end) {
    last_batch = parg->end - i;
    cur_batch = last_batch >= parg->task_batch? parg->task_batch:last_batch;

    for (j = 0; j < cur_batch; ++j) {
#ifdef _PYTHON_MBSGD
      batch_data[j] = &parg->parg_train->data[parg->index[i + j]];
      predicted = classify(batch_data[j], parg->parg_train->data_size, parg->weights, parg->parg_train->feature_size);
#else
      batch_data[j] = parg->parg_train->data[parg->index[i + j]];
      predicted = classify(batch_data[j], parg->weights, parg->parg_train->feature_size);
#endif
      z[j] = predicted - parg->parg_train->labels[parg->index[i + j]];
    }
    parg->mu += parg->y3;
    for (j = 0; j < parg->parg_train->feature_size; ++j) {
      pd = 0;
      for (k = 0; k < cur_batch; ++k) {
#ifdef _PYTHON_MBSGD
        pd += z[k] * batch_data[k][j * parg->parg_train->data_size];
#else
        pd += z[k] * batch_data[k][j];
#endif
      }
      pd /= cur_batch;
      v[j] = pd + parg->parg_train->gama * (v[j] + pd - old_pd[j]);
      old_pd[j] = pd;
      parg->weights[j] -= parg->yita * v[j];
      if (parg->parg_train->l1) {
        a = parg->weights[j];
        if(a > 0.0) {
            b = a - (parg->mu + parg->total_l1[j]);
            parg->weights[j] = b > 0.0 ? b:0.0;
        }
        else if(a < 0.0) {
            b = a + (parg->mu - parg->total_l1[j]);
            parg->weights[j] = b < 0.0 ? b:0.0;
        }
        parg->total_l1[j] += (double)(parg->weights[j] - a);
      }    
    }
    i += parg->task_batch;
  }

  free(v);
  free(old_pd);
  free(z);
  free(batch_data);

  return NULL;
}

