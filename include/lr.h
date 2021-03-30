#pragma once

float vecnorm(float *w1, float *w2, size_t size);
float l1norm(float *weights, size_t size);
float classify(float *features, float *weights, size_t feature_size);
