#pragma once
#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

class LaneDetector {
 private:
  void AbsSobelThresh(const cv::Mat image, cv::Mat& dst, const char& orient='x', const std::vector<int> threash={0,255});

  void LuvSelect(const cv::Mat& src,cv::Mat& dst,const char& channel='l',const std::vector<int> threash={0,255});

  void HsvSelect(const cv::Mat& src,cv::Mat& dst,const char& channel='s',const std::vector<int> threash={0,255});

  void LabSelect(const cv::Mat& src,cv::Mat& dst,const char& channel='b',const std::vector<int> threash={0,255});

  void RgbSelect(const cv::Mat& src,cv::Mat& channel_r,cv::Mat& channel_g,cv::Mat& channel_b,const std::vector<int> threash_r={0,255},const std::vector<int> threash_g={0,255},const std::vector<int> threash_b={0,255});

 public:
  void ColorGradientThreshold(const cv::Mat image);
};
