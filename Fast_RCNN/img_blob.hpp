#pragma once
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
using namespace caffe;
class img_blob
{
    public:
    img_blob();
    void prep_im_size();

    private:
    Blob<float> blob;
    int scales;
    cv::Size size;
    cv::Size input_geometry_;
};
