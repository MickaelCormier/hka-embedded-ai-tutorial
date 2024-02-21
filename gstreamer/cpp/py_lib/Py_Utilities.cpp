#include "Py_Utilities.h"

cv::Size2i tuple_to_size(const py::tuple& a_size_tuple) {
  if (a_size_tuple.size() != 2) {
    throw std::runtime_error("Size tuple must have exactly two elements.");
  }

  return cv::Size2i(a_size_tuple[0].cast<int>(), a_size_tuple[1].cast<int>());
}

py::tuple size_to_tuple(const cv::Size2i& a_size) {
  return py::make_tuple(a_size.width, a_size.height);
}

cv::Rect2f tuple_to_rect(const py::tuple& a_rect_tuple) {
  if (a_rect_tuple.size() != 4) {
    throw std::runtime_error("Rect tuple must have exactly four elements.");
  }

  return cv::Rect2f(
      a_rect_tuple[0].cast<float>(), a_rect_tuple[1].cast<float>(),
      a_rect_tuple[2].cast<float>(), a_rect_tuple[3].cast<float>());
}

py::tuple rect_to_tuple(const cv::Rect2f& a_rect) {
  return py::make_tuple(a_rect.x, a_rect.y, a_rect.width, a_rect.height);
}

cv::Mat numpy_to_mat(const py::array& a_input) {
  if (a_input.size() == 0) return cv::Mat();

  if (py::isinstance<py::array_t<unsigned char>>(a_input)) {
    return numpy_to_mat_int(a_input);
  } else if (py::isinstance<py::array_t<float>>(a_input)) {
    return numpy_to_mat_float(a_input);
  } else {
    throw std::runtime_error("Unsupported array type.");
  }
}

py::array mat_to_numpy(const cv::Mat& a_input) {
  if (a_input.empty()) return py::array();

  if (a_input.type() == CV_8UC1 || a_input.type() == CV_8UC3) {
    return mat_to_numpy_int(a_input);
  } else if (a_input.type() == CV_32FC1 || a_input.type() == CV_32FC3) {
    return mat_to_numpy_float(a_input);
  } else {
    throw std::runtime_error("Unsupported cv::Mat type.");
  }
}

cv::Mat numpy_to_mat_int(const py::array& a_input) {
  cv::Mat output;
  output.resize(a_input.size());

  py::buffer_info buf = a_input.request();
  if (buf.ndim != 3)
    throw std::runtime_error("Incompatible array Type: expected ndim == 3.");

  if (buf.shape[2] != 1 && buf.shape[2] != 3)
    throw std::runtime_error(
        "Incompatible array shape: expected 1 or 3 channels.");

  auto type = buf.ndim == 1 ? CV_8UC1 : CV_8UC3;
  cv::Mat mat(buf.shape[0], buf.shape[1], type, buf.ptr);

  return mat;
}

py::array mat_to_numpy_int(const cv::Mat& a_input) {
  if (a_input.type() != CV_8UC1 && a_input.type() != CV_8UC3)
    throw std::runtime_error(
        "Incompatible Mat Type: expected CV_8UC1 || CV_8UC3.");

  py::array array(py::buffer_info(
      a_input.data, sizeof(unsigned char),
      py::format_descriptor<unsigned char>::format(), a_input.channels(),
      {a_input.rows, a_input.cols, a_input.channels()},
      {sizeof(unsigned char) * a_input.cols * a_input.channels(),
       sizeof(unsigned char) * a_input.channels(), sizeof(unsigned char)}));

  return array;
}

cv::Mat numpy_to_mat_float(const py::array& a_input) {
  py::buffer_info buf = a_input.request();
  if (buf.ndim != 3)
    throw std::runtime_error("Incompatible array Type: expected ndim == 3.");

  if (buf.shape[2] != 1 && buf.shape[2] != 3)
    throw std::runtime_error(
        "Incompatible array shape: expected 1 or 3 channels.");

  auto type = buf.ndim == 1 ? CV_32FC1 : CV_32FC3;
  cv::Mat mat(buf.shape[0], buf.shape[1], type, buf.ptr);

  return mat;
}

py::array mat_to_numpy_float(const cv::Mat& a_input) {
  if (a_input.type() != CV_32FC1 && a_input.type() != CV_32FC3)
    throw std::runtime_error(
        "Incompatible Mat Type: expected CV_32FC1 || CV_32FC3.");

  py::array array(py::buffer_info(
      a_input.data, sizeof(float), py::format_descriptor<float>::format(),
      a_input.channels(), {a_input.rows, a_input.cols, a_input.channels()},
      {sizeof(float) * a_input.cols * a_input.channels(),
       sizeof(float) * a_input.channels(), sizeof(float)}));

  return array;
}
