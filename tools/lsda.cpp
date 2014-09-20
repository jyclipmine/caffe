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
  void prepare_input(const Mat& im);
  void forward_net();
  void draw_result(const Mat& im, Mat& out);
  
  // Caffe Network
  shared_ptr<Net<float> > net_;

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

  // Dynamic Data (these data will be updated for each image)
  Blob<float> bottom_imagedata_;
  Blob<float> bottom_conv5_windows_;
  Blob<float> bottom_conv5_scales_;
  Blob<float> bottom_boxes_;  
  Blob<float> bottom_valid_vec_;
  Blob<float> top_result_vecs_;
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
  bottom_imagedata_.Reshape(scale_num_, 3, input_size_, input_size_);
  bottom_conv5_windows_.Reshape(1, 1, max_proposal_num_, 4);
  bottom_conv5_scales_.Reshape(1, 1, 1, max_proposal_num_);
  bottom_boxes_.Reshape(1, 1, max_proposal_num_, 4);  
  bottom_valid_vec_.Reshape(1, 1, 1, max_proposal_num_);
  top_result_vecs_.Reshape(1, 1, 1, 3*max_proposal_num_);

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
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  CHECK_EQ(input_blobs[0]->count(), bottom_imagedata_.count())
      << "input imagedata mismatch";
  CHECK_EQ(input_blobs[1]->count(), bottom_conv5_windows_.count())
      << "input conv5_windows mismatch";
  CHECK_EQ(input_blobs[2]->count(), bottom_conv5_scales_.count())
      << "input conv5_scales mismatch";
  CHECK_EQ(input_blobs[3]->count(), bottom_boxes_.count())
      << "input boxes mismatch";
  CHECK_EQ(input_blobs[4]->count(), bottom_valid_vec_.count())
      << "input valid_vec mismatch";
  CHECK_EQ(output_blobs[0]->count(), top_result_vecs_.count())
      << "output size mismatch";
}

void LSDA::run_on_image(const string& impath, const string& outpath) {
  Mat im, out;
  LOG(INFO) << "detecting image " << impath;
  im = imread(impath);
  prepare_input(im);
  forward_net();
  draw_result(im, out);
  imwrite(outpath, out);
  LOG(INFO) << "detection results written to " << outpath;
}

void LSDA::prepare_input(const Mat& im) {
  clock_t t_start, t_finish;

  // input image size
  const int im_height = im.rows;
  const int im_width = im.cols;
  const int im_length = max(im_height, im_width);

  // 1. run Geodesic Object Proposal on input image to get bounding boxes
  t_start = clock() * 1000 / CLOCKS_PER_SEC;
  float* boxes = bottom_boxes_.mutable_cpu_data();
  int proposal_num = window_proposer_->propose(im, boxes, max_proposal_num_);
  t_finish = clock() * 1000 / CLOCKS_PER_SEC;
  LOG(INFO) << "run Geodesic Object Proposal: " << t_finish - t_start << " ms"
      << ", got " << proposal_num << " candidate windows";

  // 2. convert the candidate windows on image to regions on conv5 feature map
  t_start = clock() * 1000 / CLOCKS_PER_SEC;
  float* conv5_windows = bottom_conv5_windows_.mutable_cpu_data();
  float* conv5_scales = bottom_conv5_scales_.mutable_cpu_data();
  float* valid_vec = bottom_valid_vec_.mutable_cpu_data();
  // zero out all data
  memset(conv5_windows, 0, max_proposal_num_ * 4 * sizeof(float));
  memset(conv5_scales, 0, max_proposal_num_ * sizeof(float));
  memset(valid_vec, 0, max_proposal_num_ * sizeof(float));
  // calculate the corresponing windows on conv5 feature map
  const int conv5_stride = 16;
  const float desired_area = 50176; // 224 * 224;
  for (int i = 0; i < proposal_num; i++) {
    float y1 = boxes[4*i  ], x1 = boxes[4*i+1];
    float y2 = boxes[4*i+2], x2 = boxes[4*i+3];
    // find the best matching scale
    float area = (y2 - y1 + 1) * (x2 - x1 + 1);
    float min_area_diff = FLT_MAX;
    int matching_scale = -1;
    float matching_zoom = 0;
    for (int scale = 0; scale < scale_num_; scale++) {
      int fixed_size = fixed_sizes_vec_[scale];
      const float zoom = static_cast<float>(fixed_size) / im_length;
      float zoomed_area = area * zoom * zoom;
      float area_diff = abs(zoomed_area - desired_area);
      if (area_diff < min_area_diff) {
        min_area_diff = area_diff;
        matching_scale = scale;
        matching_zoom = zoom;
      }
    }
    conv5_scales[i] = matching_scale;
    // round and add 1 to ends
    conv5_windows[4*i  ] = 
        static_cast<int>(0.5f + y1 * matching_zoom / conv5_stride);
    conv5_windows[4*i+1] =
        static_cast<int>(0.5f + x1 * matching_zoom / conv5_stride);
    conv5_windows[4*i+2] =
        static_cast<int>(0.5f + y2 * matching_zoom / conv5_stride) + 1;
    conv5_windows[4*i+3] =
        static_cast<int>(0.5f + x2 * matching_zoom / conv5_stride) + 1;
    // set valid to be true
    valid_vec[i] = 1;
  }
  t_finish = clock() * 1000 / CLOCKS_PER_SEC;
  LOG(INFO) << "map candidate windows to regions on conv5: "
      << t_finish - t_start << " ms";

  // 3. get multiscale image pyramid
  t_start = clock() * 1000 / CLOCKS_PER_SEC;
  float* imagedata = bottom_imagedata_.mutable_cpu_data();
  const int channels = 3;
  // image offset for the expanded image
  const int input_offset = input_size_ * input_size_ * channels;
  // zero out image data
  memset(imagedata, 0, input_offset * scale_num_ * sizeof(float));
  // resize image and perform mean subtraction
  Mat resized_im;
  for (int scale = 0; scale < scale_num_; scale++) {
    int fixed_size = fixed_sizes_vec_[scale];
    const float zoom = static_cast<float>(fixed_size) / im_length;
    int resized_h = im_height * zoom;
    int resized_w = im_width * zoom;
    resize(im, resized_im, Size(resized_w, resized_h));
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < resized_h; ++h) {
        for (int w = 0; w < resized_w; ++w) {
          float pixel = static_cast<float>(resized_im.at<cv::Vec3b>(h, w)[c]);
          int index = (input_offset * scale)
              + (w + input_size_ * (h + input_size_ * c));
          imagedata[index] = pixel - channel_mean_[c];
        }
      }
    }
  }
  LOG(INFO) << "get multiscale image pyramid: " << t_finish - t_start << " ms";
}

void LSDA::forward_net() {
  clock_t t_start, t_finish;

  t_start = clock() * 1000 / CLOCKS_PER_SEC;
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  // copy data to network input
  caffe_copy(input_blobs[0]->count(), bottom_imagedata_.mutable_cpu_data(),
    input_blobs[0]->mutable_gpu_data());
  caffe_copy(input_blobs[1]->count(), bottom_conv5_windows_.mutable_cpu_data(),
    input_blobs[1]->mutable_gpu_data());
  caffe_copy(input_blobs[2]->count(), bottom_conv5_scales_.mutable_cpu_data(),
    input_blobs[2]->mutable_gpu_data());
  caffe_copy(input_blobs[3]->count(), bottom_boxes_.mutable_cpu_data(),
    input_blobs[3]->mutable_cpu_data()); // copy to CPU
  caffe_copy(input_blobs[4]->count(), bottom_valid_vec_.mutable_cpu_data(),
    input_blobs[4]->mutable_cpu_data()); // copy to CPU
  // forward network
  net_->ForwardPrefilled();
  // retrieve data
  caffe_copy(output_blobs[0]->count(), output_blobs[0]->cpu_data(),
    top_result_vecs_.mutable_cpu_data());
  t_finish = clock() * 1000 / CLOCKS_PER_SEC;
  LOG(INFO) << "network forward: " << t_finish - t_start << " ms";
}

void LSDA::draw_result(const Mat& im, Mat& out) {
  const float* keep_vec = top_result_vecs_.cpu_data();
  const float* class_id_vec = keep_vec + max_proposal_num_;
  const float* score_vec = keep_vec + max_proposal_num_ * 2;
  const float* boxes = bottom_boxes_.cpu_data();
  const static CvScalar blue = cvScalar(255, 0, 0);
  const static CvScalar red = cvScalar(0, 0, 255);
  const static CvScalar white = cvScalar(255, 255, 255);
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, CV_AA);
  char label[200];
  const int strong_cls_num = 200;
  im.copyTo(out);
  for (int box_id = 0; box_id < max_proposal_num_; box_id++) {
    if (keep_vec[box_id]) {
      int y1 = static_cast<int>(boxes[box_id*4  ]);
      int x1 = static_cast<int>(boxes[box_id*4+1]);
      int y2 = static_cast<int>(boxes[box_id*4+2]);
      int x2 = static_cast<int>(boxes[box_id*4+3]);
      y1 = max(y1, 15);
      int class_id = static_cast<int>(class_id_vec[box_id]);
      float score = score_vec[box_id];
      sprintf(label, "%s: %.3f", class_name_vec_[class_id].c_str(), score);
      Point ul(x1, y1), ur(x2, y1), ll(x1, y2), lr(x2, y2);
      line(out, ul, ur, (class_id < strong_cls_num ? blue : red), 3);
      line(out, ur, lr, (class_id < strong_cls_num ? blue : red), 3);
      line(out, lr, ll, (class_id < strong_cls_num ? blue : red), 3);
      line(out, ll, ul, (class_id < strong_cls_num ? blue : red), 3);
      line(out, ul, ur, white, 1);
      line(out, ur, lr, white, 1);
      line(out, lr, ll, white, 1);
      line(out, ll, ul, white, 1);
      IplImage iplimage = out;
      cvPutText(&iplimage, label, cvPoint(x1, y1 - 3), &font,
          (class_id < strong_cls_num ? CV_RGB(0, 0, 255) : CV_RGB(255, 0, 0)));
      LOG(INFO) << "box id " << box_id << ", (x1,y1,x2,y2) = (" << x1 << ","
          << y1 << "," << x2 << "," << y2 << "), class No. " << (class_id + 1) << ": "
          << label;
    }
  }
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
