#ifndef PY_LIB_PY_UTILITIES_H_
#define PY_LIB_PY_UTILITIES_H_

#include <vector>

#include "Py_Include.h"

/**
 * @brief Converts a Python tuple to an OpenCV Size2i object.
 *
 * @param a_size_tuple The Python tuple representing size (width, height).
 * @return cv::Size2i object constructed from the tuple.
 */
cv::Size2i tuple_to_size(const py::tuple& a_size_tuple);

/**
 * @brief Converts an OpenCV Size2i object to a Python tuple.
 *
 * @param a_size The OpenCV Size2i object.
 * @return py::tuple Python tuple representing the size (width, height).
 */
py::tuple size_to_tuple(const cv::Size2i& a_size);

/**
 * @brief Converts a Python tuple to an OpenCV Rect2f object.
 *
 * @param a_rect_tuple The Python tuple representing size (x, y, width, height).
 * @return cv::Rect2f object constructed from the tuple.
 */
cv::Rect2f tuple_to_rect(const py::tuple& a_rect_tuple);

/**
 * @brief Converts an OpenCV Rect2f object to a Python tuple.
 *
 * @param a_rect The OpenCV Rect2f object.
 * @return py::tuple Python tuple representing the size (x, y, width, height).
 */
py::tuple rect_to_tuple(const cv::Rect2f& a_rect);

/**
 * @brief Converts an numpy array to a OpenCV Mat objects.
 *
 * @param a_input Numpy array.
 * @return cv::Mat cv::Mat objects.
 */
cv::Mat numpy_to_mat(const py::array& a_input);

/**
 * @brief Converts a OpenCV Mat objects to a numpy array.
 *
 * @param a_input OpenCV Mat objects.
 * @return py::array numpy array.
 */
py::array mat_to_numpy(const cv::Mat& a_input);

/**
 * @brief Converts an numpy array to a OpenCV Mat objects
 * (int type).
 *
 * @param a_input numpy array.
 * @return cv::Mat cv::Mat objects (int type).
 */
cv::Mat numpy_to_mat_int(const py::array& a_input);

/**
 * @brief Converts a OpenCV Mat objects to a numpy array
 * (int type).
 *
 * @param a_input OpenCV Mat objects.
 * @return py::array numpy array (int type).
 */
py::array mat_to_numpy_int(const cv::Mat& a_input);

/**
 * @brief Converts an numpy array to a OpenCV Mat objects
 * (float type).
 *
 * @param a_input numpy array.
 * @return cv::Mat cv::Mat objects (float type).
 */
cv::Mat numpy_to_mat_float(const py::array& a_input);

/**
 * @brief Converts a OpenCV Mat objects to a numpy array
 * (float type).
 *
 * @param a_input OpenCV Mat objects.
 * @return py::array numpy array (float type).
 */
py::array mat_to_numpy_float(const cv::Mat& a_input);

#endif  // PY_LIB_PY_UTILITIES_H_
