// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype SPPDetectorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Blob<Dtype>* window_proposals = bottom[0]->gpu_data();
  for (int i = 0; i < proposal_num_; i++) {
    // Set ROI
    // No checks here. SpatialPyramidPoolingLayer<Dtype>::setROI will check range.
    int roi_start_h = window_proposals[4*i];
    int roi_start_w = window_proposals[4*i+1];
    int roi_end_h = window_proposals[4*i+2];
    int roi_end_w = window_proposals[4*i+3];
    spp_layers_[i]->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
    // Forward
    spp_layers_[i]->Forward(spp_bottom_vecs_[i], &(spp_top_vecs_[i]));
    // Copy the data
    caffe_copy(spp_top_vecs_[i][0]->count(), spp_top_vecs_[i][0]->gpu_data(),
        (*top)[0]->mutable_gpu_data() + dim_ * i);
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
