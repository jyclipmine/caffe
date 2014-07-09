// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype SpatialPyramidPoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype loss = 0;
  if (num_pyramid_levels_ > 1) {
    split_layer_->Forward(bottom, &split_top_vec_);
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      loss += pyramid_levels_[i]->Forward(pooling_bottom_vecs_[i], &(pooling_top_vecs_[i]));
      loss += flatten_layers_[i]->Forward(pooling_top_vecs_[i],
                                          &(flatten_top_vecs_[i]));
    }
    loss += concat_layer_->Forward(concat_bottom_vec_, top);
  } else {
    loss = pyramid_levels_[0]->Forward(bottom, top);
  }
  return Dtype(0);
}

template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  if (num_pyramid_levels_ > 1) {
    concat_layer_->Backward(top, propagate_down, &concat_bottom_vec_);
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      flatten_layers_[i]->Backward(flatten_top_vecs_[i], propagate_down,
                                   &(pooling_top_vecs_[i]));
      pyramid_levels_[i]->Backward(pooling_top_vecs_[i], propagate_down,
                                   &(pooling_bottom_vecs_[i]));
    }
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  } else {
    pyramid_levels_[0]->Backward(top, propagate_down, bottom);
  }
}


INSTANTIATE_CLASS(SpatialPyramidPoolingLayer);


}  // namespace caffe
