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
Dtype SPPDetectorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* conv_windows = bottom[1]->cpu_data();
  const Dtype* conv_scales = bottom[2]->cpu_data();
  int n = 0;
  for (n = 0; n < proposal_num_; n++) {
    int roi_start_h = conv_windows[4*n];
    int roi_start_w = conv_windows[4*n+1];
    int roi_end_h = conv_windows[4*n+2];
    int roi_end_w = conv_windows[4*n+3];
    int scale = conv_scales[n];
    // an [0 0 0 0] box marks the end of all boxes
    if (!roi_start_h && !roi_start_w && !roi_end_h && !roi_end_w) {
      break;
    }
    CHECK_GE(scale, 0) << "invalid scale: " << scale << " of window " << n;
    CHECK_LT(scale, scale_num_) << "invalid scale: " << scale
        << " of window " << n;
    // Copy data into SPP net
    caffe_copy(conv5_dim_, bottom[0]->gpu_data() + conv5_dim_ * scale,
        spp_bottom_vecs_[scale][0]->mutable_gpu_data());
    // Set ROI. No checks here.
    // SpatialPyramidPoolingLayer<Dtype>::setROI will check range.
    spp_layers_[scale]->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
    // Forward
    spp_layers_[scale]->Forward(spp_bottom_vecs_[scale],
        &(spp_top_vecs_[scale]));
    // Copy data out of SPP net
    caffe_copy(spp5_dim_, spp_top_vecs_[scale][0]->gpu_data(),
        (*top)[0]->mutable_gpu_data() + spp5_dim_ * n);
  }
  LOG(INFO) << "Forwarding " << n << " boxes in this batch";
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
