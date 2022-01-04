#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "LaneDetector.hpp"

void LaneDetector::AbsSobelThresh(const cv::Mat image, cv::Mat& dst, const char& orient, const std::vector<int> threash) {

  const int soble_kernel = 15;
#if 0
  cv::Mat gray;
  cv::Mat sobel;
  cv::Mat abs_gray;

  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  if(orient=='x') {
    cv::Sobel(gray, sobel, CV_64F, 1, 0, soble_kernel);
  }

  if(orient=='y') {
    cv::Sobel(gray, sobel, CV_64F, 0, 1, soble_kernel);
  }

  cv::convertScaleAbs(sobel,abs_gray);

  cv::imshow("x",sobel);
  cv::waitKey(0);
  cv::imshow("y",abs_gray);
  cv::waitKey(0);


  cv::inRange(abs_gray,threash[0],threash[1],dst);
#endif

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Mat sobel;
  if(orient) {
    //x
    cv::Sobel(gray, sobel, CV_64F, 1, 0, soble_kernel);
  } else {
    //y
    cv::Sobel(gray, sobel, CV_64F, 0, 1, soble_kernel);
  }

  cv::normalize(sobel,sobel,1.0,0.0,cv::NORM_INF);
  cv::convertScaleAbs(sobel*255,sobel);

  cv::inRange(sobel,threash[0],threash[1],dst);

  // std::cout<< sobel <<std::endl;

  return;
}


void LaneDetector::LuvSelect(const cv::Mat& src,cv::Mat& dst,const char& channel,const std::vector<int> threash) {
  cv::Mat luv, grad;
  std::vector<cv::Mat> channels;
  cv::cvtColor(src,luv,cv::COLOR_BGR2Luv);
  cv::split(luv,channels);

  switch (channel)  {
      case 'l':
          grad=channels.at(0);
          break;
      case 'u':
          grad=channels.at(1);
          break;
      case 'v':
          grad=channels.at(2);
          break;
  }

  cv::inRange(grad,threash[0],threash[1],dst);

  return;
}


void LaneDetector::HsvSelect(const cv::Mat& src,cv::Mat& dst,const char& channel,const std::vector<int> threash) {
  cv::Mat hsv, grad;
  std::vector<cv::Mat> channels;
  cv::cvtColor(src,hsv,cv::COLOR_BGR2HSV);
  cv::split(hsv,channels);

  switch (channel)  {
      case 'h':
          grad=channels.at(0);
          break;
      case 's':
          grad=channels.at(1);
          break;
      case 'v':
          grad=channels.at(2);
          break;
  }

  cv::inRange(grad,threash[0],threash[1],dst);

  return;

}

void LaneDetector::LabSelect(const cv::Mat& src,cv::Mat& dst,const char& channel,const std::vector<int> threash){
  cv::Mat lab, grad;
  std::vector<cv::Mat> channels;
  cv::cvtColor(src,lab,cv::COLOR_BGR2Lab);
  cv::split(lab,channels);

  switch (channel)  {
      case 'l':
          grad=channels.at(0);
          break;
      case 'a':
          grad=channels.at(1);
          break;
      case 'b':
          grad=channels.at(2);
          break;
  }

  cv::inRange(grad,threash[0],threash[1],dst);

  return;

}

void LaneDetector::RgbSelect(const cv::Mat& src,cv::Mat& channel_r,cv::Mat& channel_g,cv::Mat& channel_b,const std::vector<int> threash_r,const std::vector<int> threash_g,const std::vector<int> threash_b) {

  cv::Mat grad_r,grad_g,grad_b;
  std::vector<cv::Mat> channels;
  cv::split(src,channels);

  grad_r = channels.at(2);
  grad_g = channels.at(1);
  grad_b = channels.at(0);

  cv::inRange(grad_r, threash_r[0], threash_r[1], channel_r);
  cv::inRange(grad_g, threash_g[0], threash_g[1], channel_g);
  cv::inRange(grad_b, threash_b[0], threash_b[1], channel_b);

  return;
}


void LaneDetector::ColorGradientThreshold(const cv::Mat image){

  cv::Mat abs_x;
  cv::Mat abs_y;
  cv::Mat luv_l;
  cv::Mat luv_v;
  cv::Mat hsv_s;
  cv::Mat lab_b;
  cv::Mat rgb_r;
  cv::Mat rgb_g;
  cv::Mat rgb_b;

  AbsSobelThresh(image, abs_x, 'x', {50,90});
  AbsSobelThresh(image, abs_y, 'y', {30,90});


  LuvSelect(image, luv_l, 'l', {60,255});
  LuvSelect(image, luv_v, 'v', {150,255});
  HsvSelect(image, hsv_s, 's', {70,100});
  LabSelect(image, lab_b, 'b', {50,255});

  RgbSelect(image,rgb_r,rgb_g,rgb_b,{225,255},{225,255},{0,255});

  cv::Mat imgout = (abs_x&abs_y)|(luv_l&luv_v&hsv_s&lab_b)|(rgb_r&rgb_g&rgb_b);
  //cv::Mat imgout = (abs_x&abs_y);

  cv::imshow("a",imgout);
  cv::waitKey(0);
}