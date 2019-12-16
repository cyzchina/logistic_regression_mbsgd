#pragma once

#include "py_base.h"

#include "lr.h"

typedef struct {
  PyObject_HEAD
  double cost_sec;
  size_t feature_size;
  double *sprint_weights;
} MBSGD_MODEL;

static int 
MBSGD_MODEL_Init(MBSGD_MODEL *self, PyObject *args) {
  self->cost_sec = 0;
  self->feature_size = 0;
  self->sprint_weights = NULL;

  PyErr_SetString(PyExc_Exception, "MBSGD_MODEL cannot be instantiated");
  return -1;
}

static void
MBSGD_MODEL_Destruct(MBSGD_MODEL *self) {
  if (self->cost_sec > 0) {
    free(self->sprint_weights);
  }
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
MBSGD_MODEL_Str(MBSGD_MODEL *Self) {
  static char object_name[] = "mbsgd model";
	return Py_BuildValue("s", object_name);
}

static PyObject*
MBSGD_MODEL_Repr(MBSGD_MODEL *Self) {
	return MBSGD_MODEL_Str(Self);
}

static PyObject*
MBSGD_MODEL_Predict(MBSGD_MODEL *self, PyObject *args) {
  PyObject *new_data;
  int type = 0;
  if (!PyArg_ParseTuple(args, "O|i", &new_data, &type)) {
    PyErr_SetString(PyExc_ValueError, "wrong params");
    return NULL;
  }

  DATA_TYPE new_data_type = get_data_type(new_data);
  if (DTWRONG == new_data_type) {
    PyErr_SetString(PyExc_TypeError, "wrong new data type");
    return NULL;
  }

  PyObject *new_shape = PyObject_GetAttrString(new_data, "shape");
  if (2 != PyTuple_GET_SIZE(new_shape)) {
	  Py_XDECREF(new_shape);
    PyErr_SetString(PyExc_ValueError, "new data is not two-dimenstions");
    return NULL;
  }

  size_t new_size = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(new_shape, 0));
  size_t feature_size = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(new_shape, 1));
	Py_XDECREF(new_shape);

  if (feature_size != self->feature_size) {
    PyErr_SetString(PyExc_ValueError, "new data has wrong feature size");
    return NULL;
  }

  double *data = NULL;
  PyObject *array = NULL;
  if (DATAFRAME == new_data_type) {
    array = PyObject_CallMethod(new_data, "__array__", NULL);
    if (NULL == array) {
      PyErr_SetString(PyExc_ValueError, "no array");
      return NULL;
    }
    data = (double*)PyArray_DATA((PyArrayObject*)array);
  }

  double predicted;
  size_t i;
  PyObject *predictions = PyTuple_New(new_size);
  if (0 == type) { 
    for (i = 0; i < new_size; ++i) {
      predicted = classify(&data[i], new_size, self->sprint_weights, feature_size);
      PyTuple_SET_ITEM(predictions, i, PyFloat_FromDouble(predicted));
    }
  }
  else {
    for (i = 0; i < new_size; ++i) {
      predicted = classify(&data[i], new_size, self->sprint_weights, feature_size);
      PyTuple_SET_ITEM(predictions, i, PyFloat_FromDouble(predicted > 0.5? 1.0:0.0));
    }
  }

	Py_XDECREF(array);
  return predictions;
}

static PyMemberDef MBSGD_MODEL_DataMembers[] = {
  {"cost_sec", T_DOUBLE, offsetof(MBSGD_MODEL, cost_sec), READONLY, "cost sec of training" },
	{NULL, 0, 0, 0, NULL}
};

static PyMethodDef MBSGD_MODEL_MethodMembers[] = {
  {"Predict", (PyCFunction)MBSGD_MODEL_Predict, METH_VARARGS, "Predict."},
  {NULL, NULL, 0, NULL}
};

static PyTypeObject MBSGD_MODEL_ClassInfo = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name      = "mbsgd.MBSGD_MODEL",
  .tp_basicsize = sizeof(MBSGD_MODEL),
  .tp_itemsize  = 0,
  .tp_dealloc   = (destructor)MBSGD_MODEL_Destruct,
  .tp_repr      = (reprfunc)MBSGD_MODEL_Repr,
  .tp_repr      = (reprfunc)MBSGD_MODEL_Str,
  .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc       = "MBSGD objects",
  .tp_methods   = MBSGD_MODEL_MethodMembers,
  .tp_members   = MBSGD_MODEL_DataMembers,
  .tp_init      = (initproc)MBSGD_MODEL_Init,
  .tp_new       = PyType_GenericNew,
};

