#include "fast_rcnn.hpp"
std::map<double, double> loadMap(string filename) {
  std::map<double, double> tmp;

  mat matrix;
  matrix.load(filename);
  for (int i = 0; i < matrix.n_rows; i++)
    tmp[matrix(i, 0)] = matrix(i, 1);
  return tmp;
}

Opts::Opts() {

  caffe_version = "caffe";
  gpu_id = 0;
  per_nms_topN = 6000;
  nms_overlap_thres = 0.7;
  after_nms_topN = 300;
  use_gpu = true;
  test_scales = 600;
}
detection_model::detection_model() {

  classes = {"aeroplane",   "bicycle", "bird",  "boat",      "bottle",
             "bus",         "car",     "cat",   "chair",     "cow",
             "diningtable", "dog",     "horse", "motorbike", "person",
             "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"};

  image_means(0) = 123.68;
  image_means(1) = 116.779;
  image_means(2) = 103.939;

  proposal_net_def = "/home/a/Documents/Motion_Capture/Fast_RCNN/"
                     "fastrcnn_model/faster_rcnn_final/"
                     "faster_rcnn_VOC0712_vgg_16layers/proposal_test.prototxt";
  proposal_net = "/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/"
                 "faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/"
                 "proposal_final";
  detection_net_def = "/home/a/Documents/Motion_Capture/Fast_RCNN/"
                      "fastrcnn_model/faster_rcnn_final/"
                      "faster_rcnn_VOC0712_vgg_16layers/"
                      "detection_test.prototxt";
  detection_net = "/home/a/Documents/Motion_Capture/Fast_RCNN/fastrcnn_model/"
                  "faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/"
                  "detection_final";
  is_share_feature = true;
  last_shared_layer_idx = 30;
  last_shared_layer_detection = "relu5_3";
  last_shared_output_blob_name = "conv5_3";

  conf_proposal.batch_size = 256;
  conf_proposal.bg_thresh_hi = 0.3;
  conf_proposal.bg_thresh_lo = 0;
  conf_proposal.bg_weight = 1;
  conf_proposal.drop_boxes_runoff_image = true;
  conf_proposal.feat_stride = 16;
  conf_proposal.fg_fractioan = 0.5;
  conf_proposal.image_means(0) = 123.68;
  conf_proposal.image_means(1) = 116.779;
  conf_proposal.image_means(2) = 103.939;
  conf_proposal.ims_per_batch = 1;
  conf_proposal.max_size = 1000;
  conf_proposal.rng_seed = 6;
  conf_proposal.scales = 600;
  conf_proposal.target_only_gt = true;
  conf_proposal.test_binary = false;
  conf_proposal.test_drop_boxes_runoff_image = false;
  conf_proposal.test_max_size = 1000;
  conf_proposal.test_min_box_size = 16;
  conf_proposal.test_nms = 0.3;
  conf_proposal.test_scales = 600;
  conf_proposal.use_flipped = true;
  conf_proposal.use_gpu = true;
  conf_proposal.anchors = {
      {-83, -39, 100, 56}, {-175, -87, 192, 104},  {-359, -183, 376, 200},
      {-55, -55, 72, 72},  {-119, -119, 136, 136}, {-247, -247, 264, 264},
      {-35, -79, 52, 96},  {-79, -167, 96, 184},   {-167, -343, 184, 360}};
  conf_proposal.output_width_map = loadMap("../fastrcnn_model/map1.csv");
  conf_proposal.output_height_map = loadMap("../fastrcnn_model/map2.csv");

  conf_detection.batch_size = 128;
  conf_detection.bbox_thresh = 0.5;
  conf_detection.bg_thresh_hi = 0.5;
  conf_detection.bg_thresh_lo = 0.1;
  conf_detection.fg_fractioan = 0.25;
  conf_detection.fg_thresh = 0.5;
  conf_detection.ims_per_batch = 2;
  conf_detection.image_means(0) = 123.68;
  conf_detection.image_means(1) = 116.779;
  conf_detection.image_means(2) = 103.939;
  conf_detection.max_size = 1000;
  conf_detection.rng_seed = 6;
  conf_detection.scales = 7;
  conf_detection.test_binary = 0;
  conf_detection.test_max_size = 1000;
  conf_detection.test_nms = 0.3;
  conf_detection.test_scales = 600;
  conf_detection.use_flipped = 1;
  conf_detection.use_gpu = 1;
}
NET::NET() {}
NET::NET(string net_def, string net) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  net_.reset(new Net<float>(net_def, caffe::TEST));
  net_->CopyTrainedLayersFrom(net);
}
void NET::Forward() {
  const vector<Blob<float> *> &result = net_->Forward();
  Blob<float> *result0 = result[0];
  Blob<float> *result1 = result[1];
  cv::Mat boxes_delta(result0->num() * result0->channels() * result0->width() *
                          result0->height() / 4,
                      4, CV_32FC1);
  float *p = result0->mutable_cpu_data();
  int num = 0;
  for (int i = 0; i < result0->num() * result0->channels() * result0->width() *
                          result0->height() / 4;
       i++) {
    for (int j = 0; j < 4; j++) {
      boxes_delta.at<float>(i, j) =
          result0->data_at(0, num % result0->channels(),
                           (num -
                            num / result0->channels() / result0->height() *
                                result0->channels() * result0->height()) /
                               result0->height(),
                           num / result0->channels() / result0->height());
      num++;
    }
  }
  feature_map_size=Size(result0->width(),result0-height());
}

void NET::inputMat(cv::Mat src, detection_model proposal_detection_model) {
  cv::Mat sample_float;
  src.convertTo(sample_float, CV_32FC3);
  cv::Scalar channel_mean = cv::mean(sample_float);
  cv::Mat mean = cv::Mat(src.rows, src.cols, sample_float.type(), channel_mean);
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean, sample_normalized);

  cv::Mat im;
  int im_size_min = min(src.cols, src.rows);
  int im_size_max = max(src.cols, src.rows);
  int im_scale =
      double(proposal_detection_model.conf_proposal.scales) / im_size_min;

  if (std::round(im_scale * im_size_max) >
      proposal_detection_model.conf_proposal.max_size) {
    im_scale = double(proposal_detection_model.conf_proposal.max_size);
  }
  cv::Size input_geometry_ =
      cv::Size(round(src.cols * im_scale), round(src.rows * im_scale));
  cv::resize(src, im, input_geometry_);

  resize(sample_normalized, sample_normalized, input_geometry_);
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, sample_normalized.channels(), sample_normalized.rows,
                       sample_normalized.cols);
  net_->Reshape();
  float *input_data = input_layer->mutable_cpu_data();
  vector<cv::Mat> input_channnels;
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channels(sample_normalized.rows, sample_normalized.cols, CV_32FC1,
                     input_data);
    input_channnels.push_back(channels);
    input_data += sample_normalized.rows * sample_normalized.cols;
  }
  cv::split(sample_normalized, input_channnels);
  CHECK(reinterpret_cast<float *>(input_channnels.at(0).data) ==
        net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wraping the Input layer of the "
         "network.";
}
Fast_RCNN::Fast_RCNN(cv::Mat mat_src) {
  m_src = mat_src;
  proposal_detection_model = detection_model();
  opts = Opts();
  NET rpn_net = NET(proposal_detection_model.proposal_net_def,
                    proposal_detection_model.proposal_net);
  rpn_net.inputMat(m_src, proposal_detection_model);
  cout << "rpn_net is sucessful" << endl;
  NET fast_rcnn_net = NET(proposal_detection_model.detection_net_def,
                          proposal_detection_model.detection_net);
  cout << "fast_rcnn_net is successful" << endl;
}
bool Fast_RCNN::Forward() {
  rpn_net.Forward();
  return true;
}
