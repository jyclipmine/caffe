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
  CHECK_EQ(bottom[0]->width(), bottom[0]->height())
      << "the width and height of (expanded) conv5 feature maps should be equal";
  CHECK_EQ(bottom[1]->width(), 4)
      << "conv5_windows dimension must be 1*1*n*4, where n is the number of proposals";
  CHECK_GE(bottom[1]->height(), 0)
      << "conv5_windows dimension must be 1*1*n*4, where n is the number of proposals";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "conv5_windows dimension must be 1*1*n*4, where n is the number of proposals";
  CHECK_EQ(bottom[1]->num(), 1)
      << "conv5_windows dimension must be 1*1*n*4, where n is the number of proposals";
  CHECK_EQ(bottom[1]->height(), bottom[2]->count())
      << "number of region proposals in conv5_windows doesn't match with that in conv5_scales";
  scale_num_ = bottom[0]->num();
  conv5_dim_ = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  proposal_num_ = bottom[1]->height();
  
  LOG(INFO) << "There are " << scale_num_ << " scales and " << proposal_num_
      << " window proposals (in each batch)";
    
  // Set up inner layers
  // There is one SPP layer for each scale
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param
      = layer_param.mutable_spatial_pyramid_pooling_param();
  *spatial_pyramid_pooling_param = this->layer_param_.spatial_pyramid_pooling_param();
  for (int scale = 0; scale < scale_num_; scale++) {
    vector<Blob<Dtype>*> spp_bottom(1, new Blob<Dtype>());
    vector<Blob<Dtype>*> spp_top(1, new Blob<Dtype>());
    // There is one SPP layer for each scale
    spp_bottom[0]->Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    shared_ptr<SpatialPyramidPoolingLayer<Dtype> > spp_layer(
        new SpatialPyramidPoolingLayer<Dtype>(layer_param));
    spp_layer->SetUp(spp_bottom, &spp_top);
    spp_bottom_vecs_.push_back(spp_bottom);
    spp_top_vecs_.push_back(spp_top);
    spp_layers_.push_back(spp_layer);
  }
  spp5_dim_ = spp_top_vecs_[0][0]->count();
  (*top)[0]->Reshape(proposal_num_, 1, 1, spp5_dim_);
  LOG(INFO) << "The output spp5 feature is " << spp5_dim_ << " dimensional";
}

template <typename Dtype>
Dtype SPPDetectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
    CHECK_LT(scale, scale_num_) << "invalid scale: " << scale << " of window " << n;
    // Copy data into SPP net
    caffe_copy(conv5_dim_, bottom[0]->cpu_data() + conv5_dim_ * scale,
        spp_bottom_vecs_[scale][0]->mutable_cpu_data());
    // Set ROI. No checks here. 
    // SpatialPyramidPoolingLayer<Dtype>::setROI will check range.
    spp_layers_[scale]->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
    // Forward
    spp_layers_[scale]->Forward(spp_bottom_vecs_[scale], &(spp_top_vecs_[scale]));
    // Copy data out of SPP net
    caffe_copy(spp5_dim_, spp_top_vecs_[scale][0]->cpu_data(),
        (*top)[0]->mutable_cpu_data() + spp5_dim_ * n);
  }
  LOG(INFO) << "Forwarding " << n << " boxes in this batch";
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
