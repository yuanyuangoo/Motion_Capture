#pragma once
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
using namespace caffe;
class img_blob
{
    public:

    img_blob(cv::Mat matsrc);

    private:
    Blob<float> blob;
    int scales;
    cv::Size size;
};
