#include "Py_Include.h"
#include "Py_Utilities.h"

PYBIND11_MODULE(py_cpp_gstreamer, m) {
  m.doc() =
      "A Python connection to the cpp_gstreamer lib written in C++";

  py::class_<cpp_gstreamer>(m, "cpp_gstreamer")
      .def(py::init<>())
      
      ;
}
