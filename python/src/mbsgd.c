#include <Python.h>
#include <moduleobject.h>

#include "mbsgd_object.h"
#include "mbsgd_model_object.h"

static PyModuleDef MBSGD_ModuleInfo = {
  PyModuleDef_HEAD_INIT,
  .m_name = "mbsgd",
  .m_doc  = "multi cores mini batch stochastic gradient descend",
  .m_size = -1,
};

PyMODINIT_FUNC
PyInit_mbsgd(void) {
  if (PyType_Ready(&MBSGD_ClassInfo) < 0) {
    return NULL;
  }

  if (PyType_Ready(&MBSGD_MODEL_ClassInfo) < 0) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&MBSGD_ModuleInfo);
  if (NULL == m) {
    return NULL;
  }

  Py_INCREF(&MBSGD_ClassInfo);
  PyModule_AddObject(m, "MBSGD", (PyObject*)&MBSGD_ClassInfo);

  Py_INCREF(&MBSGD_MODEL_ClassInfo);
  PyModule_AddObject(m, "MBSGD_MODEL", (PyObject*)&MBSGD_MODEL_ClassInfo);

  import_array();

  return m;
}
