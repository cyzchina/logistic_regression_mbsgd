#include "base.h"

double
vecnorm(double *w1, double *w2, size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    double minus = fabs(w1[i] - w2[i]);
    sum += minus * minus;
  }
  return sqrt(sum);
}

double
l1norm(double *weights, size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += fabs(weights[i]);
  }
  return sum;
}

double
sigmoid(double x) {
  static double overflow = 20.0;
  if (x > overflow) x = overflow;
  if (x < -overflow) x = -overflow;

  return 1.0/(1.0 + exp(-x));

  //if (x <= 0) return 0.422457 + 0.16041 * x + 0.0143332 * x * x;
  //return 0.577434 + 0.160474 * x + -0.014341 * x * x;
  //return 0.500011 + 0.15012 * x + -0.00159302 * x * x * x;
}

double
classify(double *features, double *weights, size_t feature_size) {
  double logit = 0.0;
  for (size_t i = 0; i < feature_size; ++i) {
    logit += features[i] * weights[i];
  }
  return sigmoid(logit);
}

