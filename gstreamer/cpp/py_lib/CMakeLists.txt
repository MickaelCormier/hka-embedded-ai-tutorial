cmake_minimum_required(VERSION 3.10)

project(py_cpp_gstreamer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${CMAKE_SOURCE_DIR}/library)

add_subdirectory(pybind11)

pybind11_add_module(${PROJECT_NAME} main.cpp Py_Include.h Py_Utilities.h Py_Utilities.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
  cpp_gstreamer
  ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/lib)
