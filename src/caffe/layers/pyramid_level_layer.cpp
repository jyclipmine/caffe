// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#define max(a, b) ((a < b) ? b : a)
#define min(a, b) ((a < b) ? a : b)

namespace caffe {

template <typename Dtype>
void PyramidLevelLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX PyramidLevel, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (this->layer_param_.pyramid_level_param().pool() ==
      PyramidLevelParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }
  Layer<Dtype>::SetUp(bottom, top);
  PyramidLevelParameter pyramid_level_param =
      this->layer_param_.pyramid_level_param();
  CHECK(pyramid_level_param.has_bin_num_h()
      && pyramid_level_param.has_bin_num_w())
      << "Both bin_num_h and bin_num_w are required";
  bin_num_h_ = pyramid_level_param.bin_num_h();
  bin_num_w_ = pyramid_level_param.bin_num_w();
  CHECK_GT(bin_num_h_, 0) << "Bin number cannot be zero";
  CHECK_GT(bin_num_w_, 0) << "Bin number cannot be zero";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_, bin_num_h_,
      bin_num_w_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max PyramidLevel, we will initialize the vector index part.
  if (this->layer_param_.pyramid_level_param().pool() ==
      PyramidLevelParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.reset(new Blob<int>(bottom[0]->num(), channels_,
                                 bin_num_h_, bin_num_w_));
  }
  // If stochastic PyramidLevel, we will initialize the random index part.
  if (this->layer_param_.pyramid_level_param().pool() ==
      PyramidLevelParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, bin_num_h_,
      bin_num_w_);
  }
  // Set Region of Interest
  int roi_start_h = (pyramid_level_param.has_roi_start_h() ?
      pyramid_level_param.roi_start_h() : 0);
  int roi_start_w = (pyramid_level_param.has_roi_start_w() ?
      pyramid_level_param.roi_start_w() : 0);
  int roi_end_h = (pyramid_level_param.has_roi_end_h() ?
      pyramid_level_param.roi_end_h() : height_);
  int roi_end_w = (pyramid_level_param.has_roi_end_w() ?
      pyramid_level_param.roi_end_w() : width_);
  this->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
}

template <typename Dtype>
void PyramidLevelLayer<Dtype>::setROI(int roi_start_h, int roi_start_w,
      int roi_end_h, int roi_end_w) {
  CHECK_GE(roi_start_h, 0) << "roi_start_h must >= 0";
  CHECK_GE(roi_start_w, 0) << "roi_start_w must >= 0";
  CHECK_LE(roi_end_h, height_) << "roi_end_h must <= height_";
  CHECK_LE(roi_end_w, width_) << "roi_end_w must <= width_";;
  CHECK_GT(roi_end_h, roi_start_h) << "roi_end_h must > roi_start_h";
  CHECK_GT(roi_end_w, roi_start_w) << "roi_end_w must > roi_start_w";
  roi_start_h_ = roi_start_h;
  roi_start_w_ = roi_start_w;
  roi_end_h_ = roi_end_h;
  roi_end_w_ = roi_end_w;
  bin_size_h_ = static_cast<float>(roi_end_h_ - roi_start_h_) / bin_num_h_;
  bin_size_w_ = static_cast<float>(roi_end_w_ - roi_start_w_) / bin_num_w_;
}

// TODO(Yangqing): Is there a faster way to do PyramidLevel in the channel-first
// case?
template <typename Dtype>
Dtype PyramidLevelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different PyramidLevel methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pyramid_level_param().pool()) {
  case PyramidLevelParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_->mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < bin_num_h_; ++ph) {
          for (int pw = 0; pw < bin_num_w_; ++pw) {
            int hstart = roi_start_h_ + max(floor(ph * bin_size_h_), 0);
            int wstart = roi_start_w_ + max(floor(pw * bin_size_w_), 0);
            int hend = min(roi_start_h_ + ceil((ph + 1) * bin_size_h_),
                roi_end_h_);
            int wend = min(roi_start_w_ + ceil((pw + 1) * bin_size_w_),
                roi_end_w_);
            const int pool_index = ph * bin_num_w_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += (*top)[0]->offset(0, 1);
        } else {
          mask += (*top)[0]->offset(0, 1);
        }
      }
    }
    break;
  case PyramidLevelParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PyramidLevelParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown PyramidLevel method.";
  }
  return Dtype(0.);
}

template <typename Dtype>
void PyramidLevelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different PyramidLevel methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pyramid_level_param().pool()) {
  case PyramidLevelParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_->cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < bin_num_h_; ++ph) {
          for (int pw = 0; pw < bin_num_w_; ++pw) {
            const int index = ph * bin_num_w_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PyramidLevelParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PyramidLevelParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown PyramidLevel method.";
  }
}


INSTANTIATE_CLASS(PyramidLevelLayer);


}  // namespace caffe
