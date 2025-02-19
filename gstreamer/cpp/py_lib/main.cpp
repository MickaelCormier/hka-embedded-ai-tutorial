#include "Py_Include.h"
#include "Py_Utilities.h"
#include "Cpp_Gstreamer.h"

PYBIND11_MODULE(py_cpp_gstreamer, m) {
  m.doc() =
      "A Python connection to the cpp_gstreamer lib written in C++";

  py::class_<Cpp_Gstreamer>(m, "Cpp_Gstreamer")
      .def(py::init<const std::string&>())
      .def("run", &Cpp_Gstreamer::run);
}
