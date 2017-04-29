#include "fast_rcnn.hpp"
std::map<double,double> loadMap(string filename)
{
	std::map<double,double> tmp;

	mat matrix;
	matrix.load(filename);
	for ( int i=0; i<matrix.n_rows; i++)
		tmp[matrix(i,0)]=matrix(i,1);
	return tmp;
}

Opts::Opts()
{

	caffe_version="caffe";
	gpu_id=0;
	per_nms_topN=6000;
	nms_overlap_thres=0.7;
	after_nms_topN=300;
	use_gpu=true;
	test_scales=600;
}
detection_model::detection_model()
{	

	classes=
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

	image_means(0)=123.68;
	image_means(1)=116.779;
	image_means(2)=103.939;
	
	proposal_net_def="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/proposal_test.prototxt";
	proposal_net="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/proposal_final";
	detection_net_def="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_test.prototxt";
	detection_net="/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_final";
	is_share_feature=true;
	last_shared_layer_idx=30;
	last_shared_layer_detection="relu5_3";
	last_shared_output_blob_name="conv5_3";
	
	conf_proposal.batch_size=256;
	conf_proposal.bg_thresh_hi=0.3;
	conf_proposal.bg_thresh_lo=0;
	conf_proposal.bg_weight=1;
	conf_proposal.drop_boxes_runoff_image=true;
	conf_proposal.feat_stride=16;
	conf_proposal.fg_fractioan=0.5;
	conf_proposal.image_means(0)=123.68;
	conf_proposal.image_means(1)=116.779;
	conf_proposal.image_means(2)=103.939;
	conf_proposal.ims_per_batch=1;
	conf_proposal.max_size=1000;
	conf_proposal.rng_seed=6;
	conf_proposal.scales=600;
	conf_proposal.target_only_gt=true;
	conf_proposal.test_binary=false;
	conf_proposal.test_drop_boxes_runoff_image=false;
	conf_proposal.test_max_size=1000;
	conf_proposal.test_min_box_size=16;
	conf_proposal.test_nms=0.3;
	conf_proposal.test_scales=600;
	conf_proposal.use_flipped=true;
	conf_proposal.use_gpu=true;
	conf_proposal.anchors={
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
	conf_proposal.output_width_map=loadMap("../fastrcnn_model/map1.csv");
	conf_proposal.output_height_map=loadMap("../fastrcnn_model/map2.csv");
	
	conf_detection.batch_size=128;
	conf_detection.bbox_thresh=0.5;
	conf_detection.bg_thresh_hi=0.5;
	conf_detection.bg_thresh_lo=0.1;
	conf_detection.fg_fractioan=0.25;
	conf_detection.fg_thresh=0.5;
	conf_detection.ims_per_batch=2;
	conf_detection.image_means(0)=123.68;
	conf_detection.image_means(1)=116.779;
	conf_detection.image_means(2)=103.939;
	conf_detection.max_size=1000;
	conf_detection.rng_seed=6;
	conf_detection.scales=600;
	conf_detection.test_binary=0;
	conf_detection.test_max_size=1000;
	conf_detection.test_nms=0.3;
	conf_detection.test_scales=600;
	conf_detection.use_flipped=1;
	conf_detection.use_gpu=1;
}
