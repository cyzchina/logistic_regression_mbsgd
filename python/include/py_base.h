#pragma once

#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

static const char *DATA_TYPES[] = {"DataFrame", "list", "tuple", "numpy.ndarray"};
static const unsigned short DATA_TYPE_NUM = 4;

typedef enum {
  DATAFRAME,
  LIST,
  TUPLE,
  NDARRAY,
  DTWRONG,
} DATA_TYPE;

static DATA_TYPE
get_data_type(PyObject *obj) {
  const char *data_type_name = Py_TYPE(obj)->tp_name;
  for (unsigned short i = 0; i < DATA_TYPE_NUM; ++i) {
    if (0 == strcmp(data_type_name, DATA_TYPES[i])) {
      return (DATA_TYPE)i;
    }
  } 

  return DTWRONG;
}
