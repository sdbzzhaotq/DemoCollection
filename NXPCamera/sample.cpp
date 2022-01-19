#include <opencv2/opencv.hpp>

//g++ sample.cpp -I/usr/include/opencv4/opencv -I/usr/include/opencv4 -lopencv_imgproc -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs
const int ROAD_CAMERA_ID = 4;


int main() {

  cv::Mat image;

  cv::namedWindow("Display window");

  //cv::VideoCapture cap(ROAD_CAMERA_ID);
  //cv::VideoCapture cap(ROAD_CAMERA_ID, cv::CAP_V4L2);
  //cv::VideoCapture cap(1, cv::CAP_GSTREAMER);
  //cv::VideoCapture cap("v4l2src device=/dev/video1 ! video/x-raw,width=1280,height=720 ! autovideosink sync=false", cv::CAP_GSTREAMER);
  //cv::VideoCapture cap("v4l2src device=/dev/video1 ! video/x-raw,framerate=30/1,width=1920,height=1080 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
  //cv::VideoCapture cap("v4l2src device=/dev/video1 ! video/x-raw,framerate=30/1,width=1280,height=720 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
  cv::VideoCapture cap("v4l2src device=/dev/video1 ! video/x-raw,width=1280,height=720 ! videoconvert ! appsink", cv::CAP_GSTREAMER);

  if (!cap.isOpened()) {

    std::cout << "cannot open camera";

  }

  while (true) {

    cap >> image;

    cv::imshow("Display window", image);

    cv::waitKey(25);

  }

  return 0;
}

