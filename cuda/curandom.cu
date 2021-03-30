#include "base.h"

bool
gpu_create_randoms(uint32_t *randoms, size_t size) {
  int fd_random;
  if (-1 == (fd_random = open("/dev/urandom", O_RDONLY))) {
  //if (-1 == (fd_random = open("/dev/random", O_RDONLY))) {
    printf("open /dev/urandoom error\n");
    return false;
  }
  
  read(fd_random, randoms, sizeof(uint32_t) * size);
  
  close(fd_random);
  return true;
}

void
gpu_shuffle(uint32_t *array, uint32_t *randoms, size_t n) {
  if (n < 1) {
    return;
  }

  gpu_create_randoms(randoms, n - 1);

  size_t i, j;
  uint32_t t;
  for (i = n - 1; i > 0; --i) {
    j = randoms[i] % (i + 1); 
    if (j == i) {
      continue;
    }
    t = array[j];
    array[j] = array[i];
    array[i] = t;
  }
}
