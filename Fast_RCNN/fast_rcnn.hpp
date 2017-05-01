#pragma once
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <armadillo>

using namespace std;
using namespace caffe;
using namespace arma;

class Opts {
public:
  Opts();
  string caffe_version;
  int gpu_id;
  int per_nms_topN;
  float nms_overlap_thres;
  int after_nms_topN;
  bool use_gpu;
  int test_scales;
};

class Conf_proposal {
public:
  int batch_size;
  float bg_thresh_hi;
  float bg_thresh_lo;
  float bg_weight;
  bool drop_boxes_runoff_image;
  int feat_stride;
  float fg_fractioan;
  float fg_thresh;
  cube::fixed<1, 1, 3> image_means;
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
  mat::fixed<9, 4> anchors;
  std::map<double, double> output_height_map;
  std::map<double, double> output_width_map;
};

class Conf_detection {
public:
  short batch_size;
  float bbox_thresh;
  float bg_thresh_hi;
  float bg_thresh_lo;
  float fg_fractioan;
  float fg_thresh;
  cube::fixed<1, 1, 3> image_means;
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

class detection_model {
public:
  detection_model();
  string proposal_net_def;
  string proposal_net;
  string detection_net;
  string detection_net_def;
  bool is_share_feature;
  int last_shared_layer_idx;
  string last_shared_layer_detection;
  string last_shared_output_blob_name;
  vector<string> classes;
  cube::fixed<1, 1, 3> image_means;
  Conf_detection conf_detection;
  Conf_proposal conf_proposal;
};

class NET {
public:
  NET(string net_def, string net);
  NET();
  void inputMat(cv::Mat src, detection_model proposal_detection_model);
  void Forward();

private:
  boost::shared_ptr<Net<float>> net_;
};

class Fast_RCNN {
public:
  Fast_RCNN(cv::Mat mat_src);
  bool Forward();

private:
  NET rpn_net;
  NET fast_rcnn_net;
  Opts opts;
  detection_model proposal_detection_model;
  cv::Mat m_src;
};
