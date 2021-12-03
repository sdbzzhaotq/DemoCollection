#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char** argv) {
    //read video
    cv::VideoCapture capture;
    capture.open("/dev/video0");
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    double dWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    cout << "camera width = " << dWidth << ", height = " << dHeight << endl;

    if (!capture.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera";
    }

    Mat frame;
    while (true) {
        bool bSuccess = capture.read(frame); // read a new frame from video 
        //Breaking the while loop if the frames cannot be captured
        if (bSuccess == false) {
            cout << "Video camera is disconnected" << endl;
            cin.get(); //Wait for any key press
            break;
        }

        //show the frame in the created window
        imshow("video", frame);

        if (waitKey(10) == 27) {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
    }

    return 0;
}


