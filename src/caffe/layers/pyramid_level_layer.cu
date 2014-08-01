// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define max(a, b) ((a < b) ? b : a)
#define min(a, b) ((a < b) ? a : b)

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels,
    const int height, const int width,
    const int roi_start_h, const int roi_start_w,
    const int roi_end_h, const int roi_end_w,
    const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w,
    Dtype* top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int hstart = roi_start_h + max(floor(ph * bin_size_h), 0);
    int wstart = roi_start_w + max(floor(pw * bin_size_w), 0);
    int hend = min(roi_start_h + ceil((ph + 1) * bin_size_h), roi_end_h);
    int wend = min(roi_start_w + ceil((pw + 1) * bin_size_w), roi_end_w);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * width * height;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
Dtype PyramidLevelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pyramid_level_param().pool()) {
  case PyramidLevelParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_->mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, roi_start_h_, roi_start_w_, roi_end_h_, roi_end_w_,
        bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_,
        top_data, mask, top_mask);
    break;
  case PyramidLevelParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PyramidLevelParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width,
    const int roi_start_h, const int roi_start_w,
    const int roi_end_h, const int roi_end_w,
    const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = max(floor((h - roi_start_h) / bin_size_h - 1), 0);
    int phend = min(ceil((h - roi_start_h + 1) / bin_size_h), bin_num_h);
    int pwstart = max(floor((w - roi_start_w) / bin_size_w - 1), 0);
    int pwend = min(ceil((w - roi_start_w + 1) / bin_size_w), bin_num_w);
    Dtype gradient = 0;
    int offset = (n * channels + c) * bin_num_h * bin_num_w;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * bin_num_w + pw] == h * width + w) {
            gradient += top_diff[ph * bin_num_w + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * bin_num_w + pw] == h * width + w) {
            gradient += top_diff[ph * bin_num_w + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PyramidLevelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pyramid_level_param().pool()) {
  case PyramidLevelParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, roi_start_h_, roi_start_w_, roi_end_h_, roi_end_w_,
        bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_, bottom_diff);
    break;
  case PyramidLevelParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PyramidLevelParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(PyramidLevelLayer);


}  // namespace caffe
