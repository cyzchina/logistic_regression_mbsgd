#pragma once

double vecnorm(double *w1, double *w2, size_t size);
double l1norm(double *weights, size_t size);

#ifdef _PYTHON_MBSGD
double classify(double *features, size_t data_size, double *weights, size_t feature_size);
#else
double classify(double *features, double *weights, size_t feature_size);
#endif
