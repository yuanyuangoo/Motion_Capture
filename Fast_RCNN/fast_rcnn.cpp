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
  VideoCapture video(argv[1]);
  if (!video.isOpened())
    return -1;
  while (true) {
    Mat frame;
    video >> frame;
    imshow("frame", frame);
    if (waitKey(30) >= 0)
      break;
  }
  return 0;
}
