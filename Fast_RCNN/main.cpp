#include "fast_rcnn.hpp"

int main(int argc, char** argv)
{
	//set Option
/*	NET rpn_net(proposal_detection_model.proposal_net_def,proposal_detection_model.proposal_net);
	cout<<"rpn_net is sucessful"<<endl;
	NET fast_rcnn_net(proposal_detection_model.detection_net_def,proposal_detection_model.detection_net);
	cout<<"fast_rcnn_net is successful"<<endl;
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
	
	
*/	

	Opts opts;
	detection_model proposal_detection_model;
	cv::Mat frame;
	frame=cv::imread("../Jordan.jpg");
	cv::imshow("Jordan",frame);
	cv::waitKey(0);
	return 1;
}
