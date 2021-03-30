#pragma once

float gpu_vecnorm(float *w1, float *w2, size_t size);
float gpu_l1norm(float *weights, size_t size);
float gpu_classify(float *features, float *weights, size_t feature_size);

