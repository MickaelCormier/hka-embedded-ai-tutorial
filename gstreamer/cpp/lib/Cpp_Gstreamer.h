#ifndef LIB_CPP_GSTREAMER_H_
#define LIB_CPP_GSTREAMER_H_

#include <opencv2/opencv.hpp>

class Cpp_Gstreamer
{
public:
    Cpp_Gstreamer(const std::string a_video_path);
    ~Cpp_Gstreamer();

    void run();

private:
    cv::VideoCapture m_cap;
    double m_max_fps, m_min_fps, m_avg_fps;
    std::vector<double> m_fps_list;

    bool process_frame(cv::Mat& a_frame);
    void update_fps_stats(double a_fps);
};

#endif // LIB_CPP_GSTREAMER_H_