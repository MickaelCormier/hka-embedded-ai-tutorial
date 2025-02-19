import py_cpp_gstreamer

gst = py_cpp_gstreamer.Cpp_Gstreamer("./traffic.mp4")
gst.run()
