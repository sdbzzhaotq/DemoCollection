#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;
//using namespace cv;
//using namespace chrono;

const int ROAD_CAMERA_ID = 0;
#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874

int main(int argc, char** argv) {

    //read video
    //VideoCapture capture(ROAD_CAMERA_ID, cv::CAP_V4L2);
    cv::VideoCapture capture("ns.mov");
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 853);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 20);
    capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
    capture.set(cv::CAP_PROP_FOCUS, 0);
    float ts[9] = {1.50330396, 0.0, -59.40969163,
                  0.0, 1.50330396, 76.20704846,
                  0.0, 0.0, 1.0};

    assert(capture.isOpened());

    cv::Size size(FRAME_WIDTH,FRAME_HEIGHT);
    const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);
    cv::Mat frame_mat, transformed_mat;

    while (true) {
        capture >> frame_mat;
        cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        //int transformed_size = transformed_mat.total() * transformed_mat.elemSize();
        //printf("transformed_size = %d, transformed_mat.data = %p\n",transformed_size,transformed_mat.data);
        //show the frame in the created window
        imshow("video", transformed_mat);

        if (cv::waitKey(10) == 27) {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
    }

    return 0;
}


