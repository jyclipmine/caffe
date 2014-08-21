// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForwardLayer(const int nthread, const Dtype* bottom_data,
    const Dtype* conv5_windows, const Dtype* conv5_scales,
    const int num, const int channels,
    const int height, const int width,
    const int bin_num_h, const int bin_num_w,
    const int layer_offset, const int output_dim,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthread) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int roi_start_h = conv5_windows[n*4];
    int roi_start_w = conv5_windows[n*4+1];
    int roi_end_h   = conv5_windows[n*4+2];
    int roi_end_w   = conv5_windows[n*4+3];
    if (roi_start_h || roi_start_w || roi_end_h || roi_end_w) {
      int s = conv5_scales[n];
      float bin_size_h = static_cast<float>(roi_end_h - roi_start_h) 
          / bin_num_h;
      float bin_size_w = static_cast<float>(roi_end_w - roi_start_w)
          / bin_num_w;
      int hstart = roi_start_h + max(static_cast<int>(floor(ph * bin_size_h)),
          0);
      int wstart = roi_start_w + max(static_cast<int>(floor(pw * bin_size_w)),
          0);
      int hend = min(static_cast<int>(roi_start_h
          + ceil((ph + 1) * bin_size_h)), roi_end_h);
      int wend = min(static_cast<int>(roi_start_w
          + ceil((pw + 1) * bin_size_w)), roi_end_w);
      Dtype maxval = -FLT_MAX;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = (((s * channels + c) * height + h) * width + w);
          if (bottom_data[bottom_index] > maxval) {
            maxval = bottom_data[bottom_index];
          }
        }
      }
      int top_index = ((c * bin_num_h + ph) * bin_num_w + pw) + layer_offset
          + n * output_dim;
      top_data[top_index] = maxval;
    }
  }
}

template <typename Dtype>
void SPPDetectorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* conv5_windows = bottom[1]->gpu_data();
  const Dtype* conv5_scales = bottom[2]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int num = (*top)[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int level = 0; level < level_num_; level++) {
    int bin_num_w = bin_num_w_vec_[level];
    int bin_num_h = bin_num_h_vec_[level];
    int layer_offset = layer_offset_vec_[level];
    int nthread = num * channels * bin_num_h * bin_num_w;
    MaxPoolForwardLayer<<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
    nthread, bottom_data, conv5_windows, conv5_scales, num, channels,
    height, width, bin_num_w, bin_num_h, layer_offset, output_dim_, top_data);
  }
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
