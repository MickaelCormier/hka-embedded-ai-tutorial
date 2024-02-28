#include "Cpp_Gstreamer.h"

/**
 * Implement the GStreamer pipeline here,
 * which calculates and outputs statistics about the process
 * and also returns the images so you can 
 * display it on the screen via the application.
 */

Cpp_Gstreamer::Cpp_Gstreamer(const std::string a_video_path) : m_max_fps(0), m_min_fps(0), m_avg_fps(0)
{
    m_fps_list.clear();
    m_cap.open(a_video_path);

    // With resize from GStreamer.
    // m_cap.open("filesrc location=" + a_video_path + " ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink", cv::CAP_GSTREAMER);
    // Without resizing.
    // m_cap.open("filesrc location=" + a_video_path + " ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    if (!m_cap.isOpened()) {
        std::cerr << "Error: Could not open Video." << std::endl;
    }
}

Cpp_Gstreamer::~Cpp_Gstreamer()
{
    m_cap.release();
    cv::destroyAllWindows();
}

void Cpp_Gstreamer::run() {
    cv::Mat frame;
    while (true)
    {
        if (!process_frame(frame)) break;

        if (cv::waitKey(1) == 'q') break;
    }    
}

bool Cpp_Gstreamer::process_frame(cv::Mat& a_frame) {
    double start = cv::getTickCount();

    if (!m_cap.read(a_frame)) return false;
    cv::imshow("Video", a_frame);

    double end = cv::getTickCount();
    double fps = cv::getTickFrequency() / (end - start);

    update_fps_stats(fps);

    return true;
}

void Cpp_Gstreamer::update_fps_stats(double a_fps) {
    m_fps_list.push_back(a_fps);
    if (a_fps > m_max_fps) m_max_fps = a_fps;
    if (a_fps < m_min_fps) m_min_fps = a_fps;

    double total_fps = 0;
    for (double fps : m_fps_list) {
        total_fps += fps;
    }
    m_avg_fps = total_fps / m_fps_list.size();

    std::cout << "Current FPS: " << a_fps << ", AVG FPS: " << m_avg_fps << ", Max FPS: " << m_max_fps << ", Min FPS: " << m_min_fps << std::endl;
}
