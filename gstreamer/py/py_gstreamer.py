import cv2
import time

"""
Implement the GStreamer pipeline here,
which calculates and outputs statistics about the process
and also displays the images on the screen.
"""
def main(a_video_path):
    cap = cv2.VideoCapture("", cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open video.")


if __name__ == "__main__":
    video_path = ""
    main(video_path)
