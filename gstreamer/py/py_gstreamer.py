import cv2
import time

"""
Implement the GStreamer pipeline here,
which calculates and outputs statistics about the process
and also displays the images on the screen.
"""
def main(a_video_path):
    cap = cv2.VideoCapture(a_video_path)

    # With resize from GStreamer.
    # cap = cv2.VideoCapture(f'filesrc location={a_video_path} ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)

    # Without resizing.
    # cap = cv2.VideoCapture(f'filesrc location={a_video_path} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)


    if not cap.isOpened():
        print("Error: Could not open video.")


    frame_count = 0
    total_time = 0
    fps_list = []
    max_fps = 0
    min_fps = float('inf')

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        cv2.imshow('Video', frame)

        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_list.append(fps)

        if fps > max_fps:
            max_fps = fps
        if fps < min_fps:
            min_fps = fps

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

        print(f'Current FPS: {fps:.2f}, AVG FPS: {avg_fps:.2f}, Max FPS: {max_fps:.2f}, Min FPS: {min_fps:.2f}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "traffic.mp4"
    main(video_path)
