#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"
#include "caffe/util/window_proposer.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <string>
#include <algorithm>

using namespace caffe;
using namespace cv;
using namespace std;

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The network weights.");
DEFINE_int32(gpu, 0,
    "Run in GPU mode on given device ID.");
DEFINE_string(channelmean, "channelmean.dat",
    "The path to the channel_mean file.");
DEFINE_string(classname, "classname.txt",
    "The path to the classname file");
DEFINE_int32(scale, 5,
    "Scale number in Spatial Pyramid Pooling network (must be either 1 or 5).");
DEFINE_string(impath, "input.jpg",
    "The path to the input image.");
DEFINE_string(outpath, "output.png",
    "The path to the output image.");

class LSDA {
public:
  LSDA(const string& model, const string& weights, const int gpu,
      const string& channelmean_path, const string& classname_path,
      const int scale);
  void run_on_image(const string& impath, const string& outpath);

private:
  void get_boxes(const Mat& im);
  void get_multiscale_conv5(const Mat& im);
  void get_multiscale_image_data(const Mat& im);
  void forward_net();
  void draw_and_save_result(const Mat& im, Mat& out);
  
  // Caffe Network
  shared_ptr<Net<float> > net_;
  Blob<float> bottom_data_;
  Blob<float> bottom_boxes_;
  Blob<float> bottom_conv5_windows_;
  Blob<float> bottom_conv5_scales_;
  Blob<float> bottom_valid_vec_;
  Blob<float> top_result_;

  // Scale Parameters
  int scale_num_;
  int input_size_;
  vector<int> fixed_sizes_vec_;
  
  // Window Proposal Method
  shared_ptr<WindowProposer> window_proposer_;
  int max_proposal_num_;
  
  // Other Data
  vector<string> class_name_vec_;
  float channel_mean_[3];
};

LSDA::LSDA(const string& model, const string& weights, const int gpu,
    const string& channelmean_path, const string& classname_path,
    const int scale) {  
  // Set up Scale Parameters
  switch (scale) {
  case 5:
    fixed_sizes_vec_.push_back(640);
    fixed_sizes_vec_.push_back(768);
    fixed_sizes_vec_.push_back(917);
    fixed_sizes_vec_.push_back(1152);
    fixed_sizes_vec_.push_back(1600);
    break;
  case 1:
    fixed_sizes_vec_.push_back(917);
    break;
  default:
    LOG(FATAL) << "Invalid scale: " << scale
        << ". scale must be either 1 or 5";
    break;
  }
  scale_num_ = scale;
  input_size_ = fixed_sizes_vec_.back();

  // Set up Window Proposal Method
  window_proposer_.reset(new GOPWindowProposer());
  max_proposal_num_ = 1000;

  // Set up blob size
  bottom_data_.Reshape(scale_num_, 3, input_size_, input_size_);
  bottom_boxes_.Reshape(1, 1, max_proposal_num_, 4);
  bottom_conv5_windows_.Reshape(1, 1, max_proposal_num_, 4);
  bottom_conv5_scales_.Reshape(1, 1, 1, max_proposal_num_);
  bottom_valid_vec_.Reshape(1, 1, 1, max_proposal_num_);
  top_result_.Reshape(1, 1, 1, 3*max_proposal_num_);

  // Load channel mean
  const int channels = 3;
  ifstream fin_cmean(channelmean_path.c_str(), ios::binary);
  CHECK(fin_cmean) << "cannot open channel mean file";
  fin_cmean.read(reinterpret_cast<char*>(channel_mean_),
      channels * sizeof(float));
  CHECK(fin_cmean) << "error reading channel mean file";
  fin_cmean.close();

  // Load class names
  const int class_num = 7604;
  class_name_vec_.resize(class_num);
  ifstream fin_cname(classname_path.c_str());
  CHECK(fin_cname) << "cannot open class name file";
  for (int i = 0; i < class_num; i++) {
    getline(fin_cname, class_name_vec_[i]);
    LOG(INFO) << "class No. " << (i+1) << ": " << class_name_vec_[i];
  }
  CHECK(fin_cname) << "error reading class name file";
  fin_cname.close();

  // Set up Caffe Network
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(gpu);
  net_.reset(new Net<float>(model.c_str()));
  net_->CopyTrainedLayersFrom(weights.c_str());
}

void LSDA::run_on_image(const string& impath, const string& outpath) {
  Mat im = imread(impath);
  LOG(INFO) << "detecting image " << impath;

  clock_t t_start, t_finish;

  // run Geodesic Object Proposal on input image to get bounding boxes
  t_start = clock() * 1000 / CLOCKS_PER_SEC;
  float* boxes = bottom_boxes_.mutable_cpu_data();
  int proposal_num = window_proposer_.propose(im, boxes, max_proposal_num_);
  t_finish = clock() * 1000 / CLOCKS_PER_SEC;
  LOG(INFO) << "run Geodesic Object Proposal: " << t_finish - t_start << " ms";

  

  // write result image to disk
  Mat out = im;
  imwrite(outpath, out);
  LOG(INFO) << "detection results written to " << impath;
}

int main(int argc, char* argv[]) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // parse input command
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  // initialize LSDA
  LSDA lsda(FLAGS_model, FLAGS_weights, FLAGS_gpu, FLAGS_channelmean,
      FLAGS_classname, FLAGS_scale);

  // run LSDA on images
  lsda.run_on_image(FLAGS_impath, FLAGS_outpath);
}
