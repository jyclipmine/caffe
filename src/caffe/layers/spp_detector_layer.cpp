// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void SPPDetectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->width(), bottom[0]->height())
      << "the width and height of (expanded) "
      << "conv5 feature maps should be equal";
  CHECK_EQ(bottom[1]->width(), 4)
      << "conv5_windows dimension must be 1*1*n*4, "
      << "where n is the number of proposals";
  CHECK_GE(bottom[1]->height(), 0)
      << "conv5_windows dimension must be 1*1*n*4, "
      << "where n is the number of proposals";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "conv5_windows dimension must be 1*1*n*4, "
      << "where n is the number of proposals";
  CHECK_EQ(bottom[1]->num(), 1)
      << "conv5_windows dimension must be 1*1*n*4, "
      << "where n is the number of proposals";
  CHECK_EQ(bottom[1]->height(), bottom[2]->count())
      << "number of region proposals in conv5_windows "
      << "doesn't match with that in conv5_scales";
  scale_num_ = bottom[0]->num();
  proposal_num_ = bottom[1]->height();
  LOG(INFO) << "There are " << scale_num_ << " scales and " << proposal_num_
      << " window proposals (in each batch)";
  level_num_ =
      this->layer_param_.spatial_pyramid_pooling_param().spatial_bin_size();
  CHECK_GT(level_num_, 0)
      << "the number of spatial pyramid level must be positive";
  CHECK_EQ(this->layer_param_.pooling_param().pool(),
      SpatialPyramidPoolingParameter_PoolMethod_MAX)
      << "only max pooling is allowed";

  // get bin numbers and calculate channel offsets
  int channels = bottom[0]->channels();
  int layer_offset = 0;
  for (int level = 0; level < level_num_; level++) {
    // for now, the bin_num_h and bin_num_w are the same
    int bin_num_h =
        this->layer_param_.spatial_pyramid_pooling_param().spatial_bin(level);
    int bin_num_w =
        this->layer_param_.spatial_pyramid_pooling_param().spatial_bin(level);
    bin_num_h_vec_.push_back(bin_num_h);
    bin_num_w_vec_.push_back(bin_num_w);
    layer_offset_vec_.push_back(layer_offset);
    layer_offset += bin_num_h * bin_num_w * channels;
  }
  output_dim_ = layer_offset;
  (*top)[0]->Reshape(proposal_num_, output_dim_, 1, 1);
}

template <typename Dtype>
void SPPDetectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* conv5_windows = bottom[1]->cpu_data();
  const Dtype* conv5_scales = bottom[2]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int num = (*top)[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int level = 0; level < level_num_; level++) {
    int bin_num_w = bin_num_w_vec_[level];
    int bin_num_h = bin_num_h_vec_[level];
    int layer_offset = layer_offset_vec_[level];
    for (int n = 0; n < num; ++n) {
      int roi_start_h = conv5_windows[n*4];
      int roi_start_w = conv5_windows[n*4+1];
      int roi_end_h   = conv5_windows[n*4+2];
      int roi_end_w   = conv5_windows[n*4+3];
      if (roi_start_h || roi_start_w || roi_end_h || roi_end_w) {
        int s = conv5_scales[n];
        float bin_size_h =
            static_cast<float>(roi_end_h - roi_start_h) / bin_num_h;
        float bin_size_w =
            static_cast<float>(roi_end_w - roi_start_w) / bin_num_w;
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < bin_num_h; ++ph) {
            for (int pw = 0; pw < bin_num_w; ++pw) {
              int hstart = roi_start_h + std::max<int>(floor(ph * bin_size_h),
                  0);
              int wstart = roi_start_w + std::max<int>(floor(pw * bin_size_w),
                  0);
              int hend = std::min<int>(roi_start_h
                  + ceil((ph + 1) * bin_size_h), roi_end_h);
              int wend = std::min<int>(roi_start_w
                  + ceil((pw + 1) * bin_size_w), roi_end_w);
              int top_index = ((c * bin_num_h + ph) * bin_num_w + pw)
                  + layer_offset + n * output_dim_;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  int bottom_index =
                      (((s * channels + c) * height + h) * width + w);
                  if (bottom_data[bottom_index] > top_data[top_index]) {
                    top_data[top_index] = bottom_data[bottom_index];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
