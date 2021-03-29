#pragma once

#ifdef _CUDA
float vecnorm(float *w1, float *w2, size_t size);
float l1norm(float *weights, size_t size);
float classify(float *features, float *weights, size_t feature_size);
#else
double vecnorm(double *w1, double *w2, size_t size);
double l1norm(double *weights, size_t size);
double classify(double *features, double *weights, size_t feature_size);
#endif
