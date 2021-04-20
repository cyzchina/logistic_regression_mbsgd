#include "base.h"
#include "lr.h"

void*
task(void *param) {
  TASK_ARG *parg = (TASK_ARG*)param;

  uint32_t last_batch, cur_batch;
  size_t i, j, k;

  float predicted, a, b, pd;
  memset(parg->old_pd, 0, sizeof(float) * parg->parg_train->feature_size);
  memset(parg->v, 0, sizeof(float) * parg->parg_train->feature_size);

  i = parg->start;
  while (i < parg->end) {
    parg->mu += parg->y3;
    last_batch = parg->end - i;
    cur_batch = last_batch >= parg->task_batch? parg->task_batch:last_batch;

    for (j = 0; j < cur_batch; ++j) {
      parg->batch_data[j] = parg->parg_train->data[parg->index[i + j]];
      predicted = classify(parg->batch_data[j], parg->weights, parg->parg_train->feature_size);
      parg->z[j] = predicted - parg->parg_train->labels[parg->index[i + j]];
    }

    for (j = 0; j < parg->parg_train->feature_size; ++j) {
      pd = 0;
      for (k = 0; k < cur_batch; ++k) {
        pd += parg->z[k] * parg->batch_data[k][j];
      }
      pd /= cur_batch;

      parg->v[j] = pd + parg->parg_train->gama * (parg->v[j] + pd - parg->old_pd[j]);
      parg->old_pd[j] = pd;
      parg->weights[j] -= parg->yita * parg->v[j];
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
        //parg->total_l1[j] += (double)(parg->weights[j] - a);
        parg->total_l1[j] += parg->weights[j] - a;
      }    
    }
    i += cur_batch;
  } 
  return NULL;
}

