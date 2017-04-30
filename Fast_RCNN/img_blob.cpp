#include "img_blob.hpp"
#include "fast_rcnn.hpp"
img_blob::img_blob(cv::Mat matsrc)
{
    m_src = matsrc;
    size=matsrc.size();
    cv::Mat sample_float;
    matsrc.convertTo(sample_float,CV_32FC3);
    cv::Scalar channel_mean=cv::mean(sample_float);
   cv::Mat mean=cv::Mat(matsrc.rows,matsrc.cols,sample_float.type(),channel_mean);
   cv::Mat sample_normalized;
   cv::subtract(sample_float,mean,sample_normalized);

}
void img_blob::prep_im_size()
{
    int im_size_min=min(size.height,size.width);
    int im_size_max=max(size.height,size.width);
    scales=double(proposal_detection_model.conf_proposal.scales)/im_size_min;
    if (round(scales*im_size_max)>proposal_detection_model.conf_proposal.max_size)
    {
        scales=double(proposal_detection_model.conf_proposal.max_size);
    }
    input_geometry_=cv::Size(round(size.width*scales),round(size.height*scales));
    resize(m_src,im,input_geometry_);
}