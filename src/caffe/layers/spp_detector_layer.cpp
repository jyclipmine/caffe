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
  
  CHECK_EQ(bottom[bottom.size() - 2]->width(), 4) << "proposal dimension must be 1*1*n*4";
  CHECK_GE(bottom[bottom.size() - 2]->height(), 0) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[bottom.size() - 2]->channels(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[bottom.size() - 2]->num(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[bottom.size() - 2]->height(), bottom[bottom.size() - 1]->count())
      << "numbers of region proposal don't match";
  scale_num_ = bottom.size() - 2;
  proposal_num_ = bottom[bottom.size() - 2]->height();
    
  // Set up inner layers
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param
      = layer_param.mutable_spatial_pyramid_pooling_param();
  *spatial_pyramid_pooling_param = this->layer_param_.spatial_pyramid_pooling_param();
  for (int scale = 0; scale < scale_num_; scale++) {
    vector<Blob<Dtype>*> spp_bottom(1, new Blob<Dtype>());
    vector<Blob<Dtype>*> spp_top(1, new Blob<Dtype>());
    spp_bottom[0]->ReshapeLike(*(bottom[scale]));
    spp_bottom[0]->ShareData(*(bottom[scale]));
    shared_ptr<SpatialPyramidPoolingLayer<Dtype> > spp_layer(
        new SpatialPyramidPoolingLayer<Dtype>(layer_param));
    spp_layer->SetUp(spp_bottom, &spp_top);
    spp_bottom_vecs_.push_back(spp_bottom);
    spp_top_vecs_.push_back(spp_top);
    spp_layers_.push_back(spp_layer);
  }
  dim_ = spp_top_vecs_[0][0]->count();
  (*top)[0]->Reshape(proposal_num_, 1, 1, dim_);
}

template <typename Dtype>
Dtype SPPDetectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* conv_windows = bottom[scale_num_]->cpu_data();
  const Dtype* conv_scales = bottom[scale_num_+1]->cpu_data();
  for (int i = 0; i < proposal_num_; i++) {
    int roi_start_h = conv_windows[4*i];
    int roi_start_w = conv_windows[4*i+1];
    int roi_end_h = conv_windows[4*i+2];
    int roi_end_w = conv_windows[4*i+3];
    int scale = conv_scales[i];
    // an [0 0 0 0] box marks the end of all boxes
    if (!roi_start_h && !roi_start_w && !roi_end_h && !roi_end_w) {
      LOG(INFO) << "Forwarding only " << i << " boxes";
      break;
    }
    CHECK_GE(scale, 0) << "invalid scale: " << scale << " of window " << i;
    CHECK_LT(scale, scale_num_) << "invalid scale: " << scale << " of window " << i;
    
    // Set ROI. No checks here. 
    // SpatialPyramidPoolingLayer<Dtype>::setROI will check range.
    spp_layers_[scale]->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
    // Forward
    spp_layers_[scale]->Forward(spp_bottom_vecs_[scale], &(spp_top_vecs_[scale]));
    // Copy the data
    caffe_copy(dim_, spp_top_vecs_[scale][0]->cpu_data(),
        (*top)[0]->mutable_cpu_data() + dim_ * i);
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
