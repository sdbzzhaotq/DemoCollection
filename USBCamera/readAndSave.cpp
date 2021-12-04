#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include "MatIO.h"

const int ROAD_CAMERA_ID = 0;
#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874

void write_mb() {
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
    char szFilename[20] = {};
    std::string name;
    static int index = 0;
    for(;;) {
        capture >> frame_mat;
        cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        sprintf(szFilename,"mb/%d.mb",index);
        name.assign(szFilename);
        Utils::write(name,transformed_mat);
        cv::imshow("video", transformed_mat);
        index++;
        if (cv::waitKey(10) == 27) {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }
}

void read_mb() {
    for(auto i=0;i<8991;i++) {
        char szFilename[20] = {};
        std::string name;
        sprintf(szFilename,"mb/%d.mb",i);
        name.assign(szFilename);
        cv::Mat frame_mat = Utils::read(name);
        cv::imshow("video", frame_mat);
        if (cv::waitKey(10) == 27) {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("argument is error, write or read\n");
        return 0;
    }

    std::string arg = argv[1];
    if("read" == arg){
        read_mb();
    } else if("write" == arg){
        write_mb();
    } else {
        printf("argument is error, write or read\n");
    }

    return 0;
}
