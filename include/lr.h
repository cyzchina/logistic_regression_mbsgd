#pragma once

double vecnorm(double *w1, double *w2, size_t size);
double l1norm(double *weights, size_t size);
double classify(double *features, double *weights, size_t feature_size);
