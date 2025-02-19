#include <Cpp_Gstreamer.h>

int main(int argc, char *argv[])
{
    Cpp_Gstreamer gst("./traffic.mp4");
    gst.run();

    return 0;
}
