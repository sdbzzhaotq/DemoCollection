#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "LaneDetector.hpp"


/**
 *@brief Function main that runs the main algorithm of the lane detection.
 *@brief It will read a video of a car in the highway and it will output the
 *@brief same video but with the plotted detected lane
 *@param argv[] is a string to the full path of the demo video
 *@return flag_plot tells if the demo has sucessfully finished
 */
int main(int argc, char *argv[]) {
    if (argc != 2) {
      std::cout << "Not enough parameters" << std::endl;
      return -1;
    }

    // The input argument is the location of the video
    std::string source = argv[1];

    cv::Mat frame;
    cv::Mat img_denoise;
    cv::Mat img_edges;
    cv::Mat img_mask;
    cv::Mat img_lines;
    std::vector<cv::Vec4i> lines;
    std::vector<std::vector<cv::Vec4i> > left_right_lines;
    std::vector<cv::Point> lane;
    std::string turn;
    LaneDetector lanedetector;  // Create the class object

    frame = cv::imread(source);
  
    lanedetector.ColorGradientThreshold(frame);

    cv::waitKey(0);

}
