#include <sys/time.h>
#include <sys/sysinfo.h>

#include "base.h"

#ifdef _CUDA
#include "cutrain.h"
#else
#include "train.h"
#endif

#include "lr.h"

static const size_t BUF_SIZE = 4096;
static const size_t ACC_DATA_SIZE = 1000;
static const unsigned int MAX_FEATURE_SIZE = 1000;

void
usage(const char *prog) {
  printf("Read training data then classify test data using logistic regression:\n");
  printf("Usage:\n");
  printf("%s [options] [training_data]\n\n", prog);
  printf("Options:\n");
  printf("-s <int>   Shuffle dataset after each iteration. default 1\n");    
  printf("-i <int>   Maximum iterations. default 500\n");   
  printf("-e <float> Convergence rate. default 0.005\n");    
  printf("-a <float> Learning rate. default 0.001\n"); 
  printf("-l <float> L1 regularization weight. default 0.1\n"); 
  printf("-t <file>  Test file to classify\n");     
  printf("-r <float> Randomise weights between -1 and 1, otherwise 0\n");    
  #ifndef _CUDA
  printf("-c <int>   cpu cores. default 4\n");    
  #endif
}

bool
read_data(const char *data_file, float **pplabels, float ***pppdata, size_t *pdata_size, size_t *pfeature_size) {
  char buf[BUF_SIZE];
  //size_t read_size = readlink(data_file, buf, BUF_SIZE);
  //if (read_size > 0) {
  //  buf[read_size] = 0;
  //  data_file = buf;
  //}

  int fd;
  if (-1 == (fd = open(data_file, O_RDONLY))) {
    printf("open %s error\n", data_file);
    return false;
  }

  *pdata_size = 0;

  size_t feature_size = *pfeature_size? *pfeature_size:MAX_FEATURE_SIZE;

  float *features = (float*)calloc(feature_size, sizeof(float));
  float label = 0;

  unsigned int available_data_size = 0;

  bool overflow = false;
  size_t col_idx = 0;
  size_t i, j = 0;
  size_t idx = 0;
  size_t read_size, total_size;

  while ((read_size = read(fd, &buf[idx], BUF_SIZE - idx)) > 0) {
    total_size = idx + read_size;
    for (i = idx; i < total_size; ++i) {
      if (overflow) {
        if ('\n' == buf[i]) {
          j = i + 1;
          overflow = false;
        }
        continue;
      }
      //if (' ' == buf[i]) {
      if (',' == buf[i]) {
        if (0 == col_idx) {
          if ('1' == buf[j] || '+' == buf[j]) {
            label = 1;
          }
          else {
            label = 0;
          }
          j = i + 1;
          col_idx = 1;
          continue;
        }

        if (col_idx >= feature_size) {
          printf("overflow\n");
          memset(features, 0, sizeof(float) * feature_size);
          col_idx = 0;
          overflow = true;
          continue;
        }

        buf[i] = 0;
        features[col_idx - 1] = atof(&buf[j]);
        //features[col_idx - 1] = atof(&(strchr(&buf[j], ':')[1]));
        ++col_idx;
        j = i + 1;
      }
      else if ('\n' == buf[i]) {
        buf[i] = 0;
        features[col_idx - 1] = atof(&buf[j]);
        //if (col_idx > feature_size) {
        //  feature_size = col_idx;
        //}

        if (available_data_size <= 0) {
          available_data_size = ACC_DATA_SIZE;
          *pppdata = (float**)realloc(*pppdata, sizeof(float*) * (*pdata_size + ACC_DATA_SIZE)); 
          *pplabels = (float*)realloc(*pplabels, sizeof(float) * (*pdata_size + ACC_DATA_SIZE)); 
        }
        (*pplabels)[*pdata_size] = label;

        if (0 == *pfeature_size) {
          feature_size = *pfeature_size = col_idx;
          (*pppdata)[*pdata_size] = (float*)realloc(features, sizeof(float) * feature_size);
        }
        else {
          (*pppdata)[*pdata_size] = features;
        }
        features = (float*)calloc(feature_size, sizeof(float));

        col_idx = 0;
        j = i + 1;

        --available_data_size;
        ++(*pdata_size);

        //memset(features, 0, sizeof(double) * MAX_FEATURE_SIZE);
      }
    }
    if (j < total_size) {
      idx = total_size - j;
      memcpy(buf, &buf[j], idx);
    } 
    else {
      idx = 0;
    }
    j = 0;
  }
  close(fd);
  free(features);

  //if (0 == *pfeature_size) {
  //  *pfeature_size = feature_size;
  //}
  return true;
} 

void
model_evaluation_index(const char *test_file, size_t feature_size, float *weights) {
  size_t i, j;
  uint32_t sparsity = 0;
  for (i = 0; i < feature_size; ++i) {
    if (fabs(weights[i]) > 1e-5) {
      ++sparsity;
    }
  }

  printf("# sparsity:    %1.4f (%d/%lu)\n", (float)sparsity / feature_size, sparsity, feature_size);

  float **test_data = NULL;
  float *test_labels = NULL;

  size_t test_data_size = 0;
  size_t test_feature_size = feature_size;
  if (!read_data(test_file, &test_labels, &test_data, &test_data_size, &test_feature_size)) {
    return;
  }

  printf("\n# classifying\n");

  size_t t_idx = -1, f_idx = test_data_size;

  float *ary_predicted = calloc(test_data_size, sizeof(float));
  float predicted;

  double tp = 0, fp = 0, tn = 0, fn = 0;

  for (i = 0; i < test_data_size; ++i) {
    predicted = classify(test_data[i], weights, test_feature_size);
    if (predicted >= 0.5) {
      if (1.0 == test_labels[i]) {
        ++tp;
        ary_predicted[++t_idx] = predicted;
      }
      else{
        ++fp;
        ary_predicted[--f_idx] = predicted;
      }
    }
    else if (1.0 == test_labels[i]) {
      ++fn;
      ary_predicted[++t_idx] = predicted;
    }
    else {
      ++tn;
      ary_predicted[--f_idx] = predicted;
    }
  }

  double auc_numerator = 0;  
  for (i = 0; i < f_idx; ++i) {
    for(j = f_idx; j < test_data_size; ++j) {
      if (ary_predicted[i] > ary_predicted[j]) {
        auc_numerator += 1;
      }
      else if (fabs(ary_predicted[i] - ary_predicted[j]) < 1e-5) {
        auc_numerator += 0.5;
      }
    }
  }

  free(ary_predicted);

  free(test_labels);

  for (i = 0; i < test_data_size; ++i) {
    free(test_data[i]);
  }
  free(test_data);

  printf("# accuracy:    %1.4f (%i/%i)\n", (tp + tn) / (tp + tn + fp + fn), (int)(tp + tn), (int)(tp + tn + fp + fn));
  printf("# precision:   %1.4f\n", tp / (tp + fp));
  printf("# recall:      %1.4f\n", tp / (tp + fn));
  printf("# f-score:     %1.4f\n", 2 * tp / (tp + fp + tp + fn));
  printf("# mcc:         %1.4f\n", ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
  printf("# tp:          %i\n", (int)tp);
  printf("# tn:          %i\n", (int)tn);
  printf("# fp:          %i\n", (int)fp);    
  printf("# fn:          %i\n", (int)fn);
  printf("# auc:         %1.4f\n", auc_numerator / ((tp + fn) * (tn + fp)));

  for (i = 0; i < feature_size; ++i) {
    printf("%f ", weights[i]);
  }
  printf("\n");
}

int
main (int argc, char* const argv[]) {
  // Max iterations
  unsigned int maxit = 500;

  // Shuffle data set
  int shuf = 1;

  // Verbose
  //int verbose = 0;

  // Randomise weights
  int randw = 0;

  #ifndef _CUDA
  int cpus = 4;
  #endif

  // Learning rate
  float alpha = 0.001;
  // L1 penalty weight
  float l1 = 0.0001;
  // Convergence threshold
  float eps = 0.005;

  // Read model file
  //const char *model_in = NULL;

  // Write model file
  //const char *model_out = NULL;

  // Data fiile
  const char *train_file = NULL;
  
  // Test file
  const char *test_file = NULL;
  
  // Predictions file
  //const char *predict_file = NULL;

  size_t i;
  
  int ch;

  if (argc < 2) {
    usage(argv[0]);
    exit(-1);
  }

  train_file = argv[argc - 1];
  if ('-' == train_file[0]) {
    usage(argv[0]);
    return -1;
  }

  #ifdef _CUDA
  while (-1 != (ch = getopt(argc, argv, "s:i:e:a:l:t:r:"))) {
  #else
  while (-1 != (ch = getopt(argc, argv, "s:i:e:a:l:t:r:c:"))) {
  #endif
    switch(ch) {
      case 's':
        shuf = atoi(optarg);
        break;
      case 'i':
        maxit = atoi(optarg);
        break;
      case 'e':
        eps = atof(optarg);
        break;
      case 'a':
        alpha = atof(optarg);
        break;
      case 'l':
        l1 = atof(optarg);
        break;
      case 't':
        test_file = optarg;
        //printf("%s %s\n", optarg, argv[optind]);
        break;
      case 'r':
        randw = 1;
        break;
      #ifndef _CUDA
      case 'c':
        cpus = atoi(optarg);
        break;
      #endif
      default:
        usage(argv[0]);
        return -1;
    }
  }

  #ifndef _CUDA
  int nprocs = get_nprocs();
  if (nprocs < cpus) {
    cpus = nprocs;
  }
  #endif

  float **data = NULL;
  float *labels = NULL;
  float gama = 0.9;

  size_t data_size = 0;
  size_t feature_size = 0;
  if (!read_data(train_file, &labels, &data, &data_size, &feature_size)) {
    return -1;
  }

  printf("# data_size:    %lu\n", data_size);
  printf("# feature_size: %lu\n", feature_size);
  #ifndef _CUDA
  printf("# cpus:         %d\n", cpus);
  #endif

  float *sprint_weights = (float*)calloc(feature_size, sizeof(float));

  TRAIN_ARG arg;
  #ifndef _CUDA
  arg.cpus = cpus;
  #endif
  arg.alpha = alpha;
  arg.gama = gama;
  arg.l1 = l1;
  arg.maxit = maxit;
  arg.shuf = shuf;
  arg.eps = eps;
  arg.randw = randw;
  arg.labels = labels;
  arg.data = data;
  arg.data_size = data_size;
  arg.feature_size = feature_size;
  arg.sprint_weights = sprint_weights;

  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  #ifdef _CUDA
  gpu_train(&arg);
  #else
  train(&arg); 
  #endif
  gettimeofday(&tv2, NULL);

  time_t cost_sec = tv2.tv_sec - tv1.tv_sec;
  suseconds_t cost_us =  tv2.tv_usec - tv1.tv_usec;
  if (cost_us < 0) {
    --cost_sec;
    cost_us += 1000000;
  }
  printf("# train cost:  %ld.%06ld\n", cost_sec, cost_us);

  free(labels);

  for (i = 0; i < data_size; ++i) {
    free(data[i]);
  }
  free(data);

  model_evaluation_index(test_file, feature_size, sprint_weights);

  free(sprint_weights);
  return 0;
}
