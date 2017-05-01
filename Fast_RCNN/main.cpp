#include "fast_rcnn.hpp"

int main(int argc, char **argv) {
  // set Option

  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

  cv::Mat frame;
  frame = cv::imread("../Jordan.jpg");
  Fast_RCNN detector(frame);
  cv::imshow("Jordan", frame);
  cv::waitKey(0);
  return 1;
}
