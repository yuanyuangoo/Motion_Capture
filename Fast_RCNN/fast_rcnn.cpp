#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(int argc, char **argv) {

  std::cout << "Hello World!" << endl;
  if (argc < 1) {
    std::cout << "No input" << endl;
    return -1;
  }
  std::cout << argv[1] << endl;

  VideoCapture video(0);
  if (!video.isOpened())
    return -1;
    namedWindow("Video",1);
  while (true) {
    Mat frame;
    video >> frame;
    if (!frame.empty()) {
      std::cout << "H" << endl;
      imshow("window", frame);
    }
    //   if (waitKey(30) >= 0)
    //     break;
  }
  return 0;
}