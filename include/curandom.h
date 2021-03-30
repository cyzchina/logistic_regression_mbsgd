#pragma once

bool gpu_create_randoms(uint32_t *randoms, size_t size);
void gpu_shuffle(uint32_t *array, uint32_t *randoms, size_t n);

