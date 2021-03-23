#pragma once

#include <stdio.h>
#include <sys/sysinfo.h>

#include "py_base.h"
#include "train.h"

#include "mbsgd_model_object.h"

typedef struct {
  PyObject_HEAD
  short cpus;
  DATA_TYPE train_data_type;
  unsigned int maxit; 
  double alpha;
  double gama;
  double l1;
  double eps;
  size_t train_size;
  size_t feature_size;
  double *labels;
  double *data;
  PyObject *array;
} MBSGD;

static int 
MBSGD_Init(MBSGD *self, PyObject *args, PyObject *kwargs) {
  self->cpus  = 4;
  self->maxit = 500;
  self->alpha = 0.001;
  self->l1    = 0.0001;
  self->eps   = 0.005;
  self->gama  = 0.9;
  self->array = NULL;

  static char *kwlist[] = {"train_data", "maxit", "eps", "cpus", "alpha", "l1", "gama"};

  PyObject *train_data;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ididdd", kwlist, 
                                   &train_data, &self->maxit, 
                                   &self->eps, &self->cpus, &self->alpha, 
                                   &self->l1, &self->gama)) {
    PyErr_SetString(PyExc_ValueError, "wrong params");
    return -1;
  }

  if (self->cpus < 0 || self->maxit < 0 || self->alpha < 1e-5 || self->l1 < 1e-5 || self->eps < 1e-5 || self->gama < 1e-5) {
    PyErr_SetString(PyExc_ValueError, "wrong param values");
    return -1;
  }

  self->train_data_type = get_data_type(train_data);
  if (DATAFRAME != self->train_data_type) {
    PyErr_SetString(PyExc_TypeError, "wrong train data type");
    return -1;
  }

  PyObject *train_shape = PyObject_GetAttrString(train_data, "shape");
  if (2 != PyTuple_GET_SIZE(train_shape)) {
    Py_XDECREF(train_shape);
    PyErr_SetString(PyExc_ValueError, "train data is not two-dimenstions");
    return -1;
  }

  self->train_size = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(train_shape, 0));
  self->feature_size = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(train_shape, 1)) - 1;
  Py_XDECREF(train_shape);

  if (DATAFRAME == self->train_data_type) {
    self->array = PyObject_CallMethod(train_data, "__array__", NULL);
    if (NULL == self->array) {
      PyErr_SetString(PyExc_ValueError, "no array");
      return -1;
    }
    double *raw_data = (double*)PyArray_DATA((PyArrayObject*)self->array);
    self->labels = raw_data;
    self->data = &raw_data[self->train_size];
  }

  int nprocs = get_nprocs();
  if (nprocs < self->cpus) {
    self->cpus = nprocs;
  }

  //PyObject *test = PyObject_GetAttrString(train_data, "values"); 
  ////PyArrayObject *test2 = (PyArrayObject*)PyArray_ContiguousFromObject(test, NPY_DOUBLE, 2, 2);
  ////double *array = (double*)PyArray_DATA(test2);
  ////for (size_t i = 0; i < 10; ++i) {
  ////    printf("%lf ", array[i]);
  ////}
  ////printf("\n");
  ////Py_XDECREF(test2);
  //PyObject *test2 = PyObject_GetAttrString(test, "data"); 
  //PyObject *test3 = PyObject_CallMethod(test2, "tolist", NULL);

  ////for (size_t i = 0; i < 10; ++i) {
  ////  printf("%f ", PyFloat_AsDouble(PyList_GetItem(test3, (Py_ssize_t)i)));
  ////}
  ////printf("\n");

  //PyObject *test4 = PyList_GetItem(test3, (Py_ssize_t)0);
  //PyObject *test5 = PyList_GetItem(test4, (Py_ssize_t)1);
  //const char *data_type_name = Py_TYPE(test5)->tp_name;
  //printf("%s\n", data_type_name);
  //printf("%.17g\n", PyFloat_AsDouble(test5));

  //
  //Py_XDECREF(test5);
  //Py_XDECREF(test4);
  //Py_XDECREF(test3);
  //Py_XDECREF(test2);
  //Py_XDECREF(test);

  return 0;
}

static void
MBSGD_Destruct(MBSGD *self) {
  Py_XDECREF(self->array);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
MBSGD_Str(MBSGD *Self) {
  static char object_name[] = "mbsgd";
  return Py_BuildValue("s", object_name);
}

static PyObject*
MBSGD_Repr(MBSGD *Self) {
  return MBSGD_Str(Self);
}

static PyObject*
MBSGD_Train(MBSGD *self, PyObject *args, PyObject *kwargs) {
  static char *kwlist[] = {"maxit", "eps", "cpus", "alpha", "l1", "gama"};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ididd", kwlist, 
                                   &self->maxit, &self->eps, &self->cpus, 
                                   &self->alpha, &self->l1, &self->gama)) {
    PyErr_SetString(PyExc_ValueError, "wrong params");
    return NULL;
  }

  if (self->cpus < 0 || self->maxit < 0 || self->alpha < 1e-5 || self->l1 < 1e-5 || self->eps < 1e-5 || self->gama < 1e-5) {
    PyErr_SetString(PyExc_ValueError, "wrong param values");
    return NULL;
  }

  double *sprint_weights = (double*)calloc(self->feature_size, sizeof(double));

  TRAIN_ARG arg;
  arg.cpus = self->cpus;
  arg.alpha = self->alpha;
  arg.gama = self->gama;
  arg.l1 = self->l1;
  arg.maxit = self->maxit;
  arg.shuf = 1;
  arg.eps = self->eps;
  arg.randw = 0;
  arg.labels = self->labels;
  arg.data = self->data;
  arg.data_size = self->train_size;
  arg.feature_size = self->feature_size;
  arg.sprint_weights = sprint_weights;

  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  train(&arg); 
  gettimeofday(&tv2, NULL);

  time_t cost_sec = tv2.tv_sec - tv1.tv_sec;
  suseconds_t cost_us =  tv2.tv_usec - tv1.tv_usec;
  
  Py_INCREF(&MBSGD_MODEL_ClassInfo);
  MBSGD_MODEL *mbsgd_model = (MBSGD_MODEL*)MBSGD_MODEL_ClassInfo.tp_alloc(&MBSGD_MODEL_ClassInfo, 0); 
  mbsgd_model->sprint_weights = sprint_weights;
  mbsgd_model->feature_size = self->feature_size;
  mbsgd_model->cost_sec = (double)cost_sec + (double)cost_us / 1000000;

  size_t i;
  mbsgd_model->weights = PyList_New(self->feature_size);
  for (i = 0; i < self->feature_size; ++i) {
    PyList_SET_ITEM(mbsgd_model->weights, i, PyFloat_FromDouble(sprint_weights[i]));
  }

  return (PyObject*)mbsgd_model;
}

static PyMemberDef MBSGD_DataMembers[] = {
  {NULL, 0, 0, 0, NULL}
};

static PyMethodDef MBSGD_MethodMembers[] = {
  {"Train", (PyCFunction)MBSGD_Train, METH_VARARGS|METH_KEYWORDS, "Train."},
  {NULL, NULL, 0, NULL}
};

static PyTypeObject MBSGD_ClassInfo = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name      = "mbsgd.MBSGD",
  .tp_basicsize = sizeof(MBSGD),
  .tp_itemsize  = 0,
  .tp_dealloc   = (destructor)MBSGD_Destruct,
  .tp_repr      = (reprfunc)MBSGD_Repr,
  .tp_repr      = (reprfunc)MBSGD_Str,
  .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc       = "MBSGD objects",
  .tp_methods   = MBSGD_MethodMembers,
  .tp_members   = MBSGD_DataMembers,
  .tp_init      = (initproc)MBSGD_Init,
  .tp_new       = PyType_GenericNew,
};

