#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iostream>

#include <armadillo>

using namespace std;
using namespace caffe;
using namespace arma;

class Opts
{
	public:
		string caffe_version;
		int gpu_id;
		int per_nms_topN;
		float nms_overlap_thres;
		int after_nms_topN;
		bool use_gpu;
		int test_scales;
};

class Conf_proposal
{
	public:
		int batch_size;
		float bg_thresh_hi;
		float bg_thresh_lo;
		float bg_weight;
		bool drop_boxes_runoff_image;
		int feat_stride;
		float fg_fractioan;
		float fg_thresh;
		cube::fixed<1,1,3> image_means;
		int ims_per_batch;
		int max_size;
		short rng_seed;
		short scales;
		bool target_only_gt;
		bool test_binary;
		bool test_drop_boxes_runoff_image;
		int test_max_size;
		short test_min_box_size;
		float test_nms;
		short test_scales;
		bool use_flipped;
		bool use_gpu;
		mat::fixed<9,4> anchors;
		std::map<double,double> output_height_map;
		std::map<double,double> output_width_map;
};

class Conf_detection
{
	public:
	short batch_size;
	float bbox_thresh;
	float bg_thresh_hi;
	float bg_thresh_lo;
	float fg_fractioan;
	float fg_thresh;
	cube::fixed<1,1,3> image_means;
	short ims_per_batch;
	short max_size;
	short rng_seed;
	short scales;
	bool test_binary;
	short test_max_size;
	float test_nms;
	short test_scales;
	bool use_flipped;
	bool use_gpu;
};

class detection_model
{
	public:
		string proposal_net_def;
		string proposal_net;
		string detection_net;
		string detection_net_def;
		bool is_share_feature;
		int last_shared_layer_idx;
		string last_shared_layer_detection;
		string last_shared_output_blob_name;
		vector<string> classes;
		cube::fixed<1,1,3> image_means;
		Conf_detection conf_detection;
		Conf_proposal conf_proposal;
};

Opts opts;
detection_model proposal_detection_model;

std::map<double,double> loadMap(string filename)
{
	std::map<double,double> tmp;

	mat matrix;
	matrix.load(filename);
	for ( int i=0; i<matrix.n_rows; i++)
		tmp[matrix(i,0)]=matrix(i,1);
	return tmp;
}

void init()
{	
	opts.caffe_version="caffe";
	opts.gpu_id=0;
	opts.per_nms_topN=6000;
	opts.nms_overlap_thres=0.7;
	opts.after_nms_topN=300;
	opts.use_gpu=true;
	opts.test_scales=600;

	proposal_detection_model.classes=
	{
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor"};

	proposal_detection_model.image_means(0)=123.68;
	proposal_detection_model.image_means(1)=116.779;
	proposal_detection_model.image_means(2)=103.939;
	
	proposal_detection_model.proposal_net_def="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/proposal_test.prototxt";
	proposal_detection_model.proposal_net="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/proposal_final";
	proposal_detection_model.detection_net_def="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_test.prototxt";
	proposal_detection_model.detection_net="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_final";
	proposal_detection_model.is_share_feature=true;
	proposal_detection_model.last_shared_layer_idx=30;
	proposal_detection_model.last_shared_layer_detection="relu5_3";
	proposal_detection_model.last_shared_output_blob_name="conv5_3";
	
	proposal_detection_model.conf_proposal.batch_size=256;
	proposal_detection_model.conf_proposal.bg_thresh_hi=0.3;
	proposal_detection_model.conf_proposal.bg_thresh_lo=0;
	proposal_detection_model.conf_proposal.bg_weight=1;
	proposal_detection_model.conf_proposal.drop_boxes_runoff_image=true;
	proposal_detection_model.conf_proposal.feat_stride=16;
	proposal_detection_model.conf_proposal.fg_fractioan=0.5;
	proposal_detection_model.conf_proposal.image_means(0)=123.68;
	proposal_detection_model.conf_proposal.image_means(1)=116.779;
	proposal_detection_model.conf_proposal.image_means(2)=103.939;
	proposal_detection_model.conf_proposal.ims_per_batch=1;
	proposal_detection_model.conf_proposal.max_size=1000;
	proposal_detection_model.conf_proposal.rng_seed=6;
	proposal_detection_model.conf_proposal.scales=600;
	proposal_detection_model.conf_proposal.target_only_gt=true;
	proposal_detection_model.conf_proposal.test_binary=false;
	proposal_detection_model.conf_proposal.test_drop_boxes_runoff_image=false;
	proposal_detection_model.conf_proposal.test_max_size=1000;
	proposal_detection_model.conf_proposal.test_min_box_size=16;
	proposal_detection_model.conf_proposal.test_nms=0.3;
	proposal_detection_model.conf_proposal.test_scales=600;
	proposal_detection_model.conf_proposal.use_flipped=true;
	proposal_detection_model.conf_proposal.use_gpu=true;
	proposal_detection_model.conf_proposal.anchors={
		{	  -83,  -39,  100,   56  	},
		{	 -175,  -87,  192,  104     },
		{	 -359, -183,  376,  200   	},
		{     -55,  -55,   72,   72		},
		{    -119, -119,  136,  136     },
		{	 -247, -247,  264,  264		},
		{     -35,  -79,   52,	 96  	},
		{	  -79, -167,   96,  184     },
		{    -167, -343,  184,  360     }
	};
	proposal_detection_model.conf_proposal.output_width_map=loadMap("../fastrcnn_model/map1.csv");
	proposal_detection_model.conf_proposal.output_height_map=loadMap("../fastrcnn_model/map2.csv");
	
	proposal_detection_model.conf_detection.batch_size=128;
	proposal_detection_model.conf_detection.bbox_thresh=0.5;
	proposal_detection_model.conf_detection.bg_thresh_hi=0.5;
	proposal_detection_model.conf_detection.bg_thresh_lo=0.1;
	proposal_detection_model.conf_detection.fg_fractioan=0.25;
	proposal_detection_model.conf_detection.fg_thresh=0.5;
	proposal_detection_model.conf_detection.ims_per_batch=2;
	proposal_detection_model.conf_detection.image_means(0)=123.68;
	proposal_detection_model.conf_detection.image_means(1)=116.779;
	proposal_detection_model.conf_detection.image_means(2)=103.939;
	proposal_detection_model.conf_detection.max_size=1000;
	proposal_detection_model.conf_detection.rng_seed=6;
	proposal_detection_model.conf_detection.scales=600;
	proposal_detection_model.conf_detection.test_binary=0;
	proposal_detection_model.conf_detection.test_max_size=1000;
	proposal_detection_model.conf_detection.test_nms=0.3;
	proposal_detection_model.conf_detection.test_scales=600;
	proposal_detection_model.conf_detection.use_flipped=1;
	proposal_detection_model.conf_detection.use_gpu=1;
}

class NET{
	public:
		NET(string net_def,string net);
	private:
		boost::shared_ptr< Net <float> > net_;
};
NET::NET(string net_def,string net)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	
	net_.reset(new Net<float>(net_def,caffe::TEST));
	net_->CopyTrainedLayersFrom(net);
	
}

int main(int argc, char** argv)
{
	//set Option
	init();
	NET rpn_net(proposal_detection_model.proposal_net_def,proposal_detection_model.proposal_net);
	cout<<"rpn_net is sucessful"<<endl;
	NET fast_rcnn_net(proposal_detection_model.detection_net_def,proposal_detection_model.detection_net);
	cout<<"fast_rcnn_net is successful"<<endl;
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
	return 1;
}
