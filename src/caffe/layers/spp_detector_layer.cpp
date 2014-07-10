// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
void SPPDetectorLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[1]->width(), 4) << "proposal dimension must be 1*1*n*4";
  CHECK_GE(bottom[1]->height(), 0) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->channels(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->num(), 1) << "proposal dimension must be 1*1*n*4";
  proposal_num_ = bottom[1]->height();
  
  // Set up inner layers
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param
      = layer_param.spatial_pyramid_pooling_param();
  *spatial_pyramid_pooling_param = layer_param_.spatial_pyramid_pooling_param();
  for (int i = 0; i < proposal_num_; i++) {
    vector<Blob<Dtype>*> spp_bottom(1, new Blob<Dtype>());
    vector<Blob<Dtype>*> spp_top(1, new Blob<Dtype>());
    spp_bottom[0].ShareData(bottom[0]);
    shared_ptr<SpatialPyramidPoolingLayer<Dtype> > spp_layer(
        new SpatialPyramidPoolingLayer<Dtype>(layer_param));
    spp_layer->SetUp(spp_bottom, spp_top);
    spp_bottom_vecs_.push_back(spp_bottom);
    spp_top_vecs_.push_back(spp_top);
    spp_layers_.push_back(spp_layer);
  }
  dim_ = spp_top_vecs_[0][0]->count();
  (*top)[0]->Reshape(1, 1, proposal_num_, dim_);
}

template <typename Dtype>
Dtype SPPDetectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const Blob<Dtype>* window_proposals = bottom[0];
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
    caffe_copy(dim_, spp_top_vecs_[i][0]->cpu_data(),
        (*top)[0]->mutable_cpu_data() + dim_ * i);
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
