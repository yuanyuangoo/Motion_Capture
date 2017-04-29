#include "img_blob.hpp"
img_blob::img_blob(cv::Mat matsrc)
{
   cv::Mat sample_float;
   matsrc.convertTo(sample_float,CV_32FC3);
   cv::Scalar channel_mean=cv::mean(sample_float);
   cv::Mat mean(cv::Mat matsrc.rows,matsrc.cols,sample_float.type(),channel_mean);
   cv::Mat sample_normalized;
   cv::subtract(sample_float,mean,sample_normalized);

}
