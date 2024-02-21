cmake_minimum_required(VERSION 3.16)

project(cpp_gstreamer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_search_module(GLIB2 REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GIO REQUIRED IMPORTED_TARGET gio-2.0)
pkg_search_module(GOB REQUIRED IMPORTED_TARGET gobject-2.0)
pkg_search_module(GSTB REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_search_module(GSTVID REQUIRED IMPORTED_TARGET gstreamer-video-1.0)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)

include_directories(${GST_INCLUDE_DIRS})
link_directories(${GST_LIBARARY_DIRS})

add_library(${PROJECT_NAME} SHARED
  cpp_gstreamer.h
  cpp_gstreamer.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    PkgConfig::GST
    PkgConfig::GIO
    PkgConfig::GOB
    PkgConfig::GSTB
    PkgConfig::GSTVID
    PkgConfig::GSTAPP
)
