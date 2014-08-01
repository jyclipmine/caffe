// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  num_pyramid_levels_ =
      this->layer_param_.spatial_pyramid_pooling_param().spatial_bin_size();
  CHECK_GE(num_pyramid_levels_, 1);

  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_concat_dim(1);
  PyramidLevelParameter* pyramid_level_param =
      layer_param.mutable_pyramid_level_param();
  switch (this->layer_param_.pooling_param().pool()) {
  case SpatialPyramidPoolingParameter_PoolMethod_MAX:
    pyramid_level_param->set_pool(PyramidLevelParameter_PoolMethod_MAX);
    break;
  case SpatialPyramidPoolingParameter_PoolMethod_AVE:
    pyramid_level_param->set_pool(PyramidLevelParameter_PoolMethod_AVE);
    break;
  case SpatialPyramidPoolingParameter_PoolMethod_STOCHASTIC:
    pyramid_level_param->set_pool(PyramidLevelParameter_PoolMethod_STOCHASTIC);
    break;
  default:
    LOG(FATAL) << "Unknown spatial pyramid pooling method " <<
      this->layer_param_.pooling_param().pool();
  }

  if (num_pyramid_levels_ > 1) {
    split_top_vec_.clear();
    split_top_vec_.push_back(bottom[0]);
    for (int i = 1; i < num_pyramid_levels_; ++i) {
      split_top_vec_.push_back(new Blob<Dtype>());
    }
    split_layer_.reset(new SplitLayer<Dtype>(layer_param));
    split_layer_->SetUp(bottom, &split_top_vec_);
    pooling_bottom_vecs_.clear();
    pooling_top_vecs_.clear();
    pyramid_levels_.clear();
    flatten_top_vecs_.clear();
    flatten_layers_.clear();
    concat_bottom_vec_.clear();
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      const int spatial_bin =
          this->layer_param_.spatial_pyramid_pooling_param().spatial_bin(i);
      pyramid_level_param->set_bin_num_h(spatial_bin);
      pyramid_level_param->set_bin_num_w(spatial_bin);
      shared_ptr<PyramidLevelLayer<Dtype> > pyramid_level_layer(
          new PyramidLevelLayer<Dtype>(layer_param));
      vector<Blob<Dtype>*> pooling_layer_bottom(1, split_top_vec_[i]);
      vector<Blob<Dtype>*> pooling_layer_top(1, new Blob<Dtype>());
      pyramid_level_layer->SetUp(pooling_layer_bottom, &pooling_layer_top);
      pooling_bottom_vecs_.push_back(pooling_layer_bottom);
      pooling_top_vecs_.push_back(pooling_layer_top);
      pyramid_levels_.push_back(pyramid_level_layer);

      shared_ptr<FlattenLayer<Dtype> > flatten_layer(
          new FlattenLayer<Dtype>(layer_param));
      vector<Blob<Dtype>*> flatten_layer_top(1, new Blob<Dtype>());
      flatten_layer->SetUp(pooling_layer_top, &flatten_layer_top);
      flatten_top_vecs_.push_back(flatten_layer_top);
      flatten_layers_.push_back(flatten_layer);

      concat_bottom_vec_.push_back(flatten_layer_top[0]);
    }
    concat_layer_.reset(new ConcatLayer<Dtype>(layer_param));
    concat_layer_->SetUp(concat_bottom_vec_, top);
  } else {
    const int spatial_bin =
          this->layer_param_.spatial_pyramid_pooling_param().spatial_bin(0);
    pyramid_level_param->set_bin_num_h(spatial_bin);
    pyramid_level_param->set_bin_num_w(spatial_bin);
    shared_ptr<PyramidLevelLayer<Dtype> > layer(
          new PyramidLevelLayer<Dtype>(layer_param));
    layer->SetUp(bottom, top);
    pyramid_levels_.push_back(layer);
  }
  // Set Region of Interest
  int roi_start_h =
      (this->layer_param_.spatial_pyramid_pooling_param().has_roi_start_h() ?
      this->layer_param_.spatial_pyramid_pooling_param().roi_start_h() : 0);
  int roi_start_w =
      (this->layer_param_.spatial_pyramid_pooling_param().has_roi_start_w() ?
      this->layer_param_.spatial_pyramid_pooling_param().roi_start_w() : 0);
  int roi_end_h =
      (this->layer_param_.spatial_pyramid_pooling_param().has_roi_end_h() ?
      this->layer_param_.spatial_pyramid_pooling_param().roi_end_h() :
      bottom[0]->height());
  int roi_end_w =
      (this->layer_param_.spatial_pyramid_pooling_param().has_roi_end_w() ?
      this->layer_param_.spatial_pyramid_pooling_param().roi_end_w() :
      bottom[0]->width());
  this->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
}

template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::setROI(int roi_start_h, int roi_start_w,
      int roi_end_h, int roi_end_w) {
  // No checks here. All checks are in PyramidLevelLayer::setROI
  for (int i = 0; i < pyramid_levels_.size(); ++i) {
    pyramid_levels_[i]->setROI(roi_start_h, roi_start_w, roi_end_h, roi_end_w);
  }
}

template <typename Dtype>
Dtype SpatialPyramidPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype loss = 0;
  if (num_pyramid_levels_ > 1) {
    split_layer_->Forward(bottom, &split_top_vec_);
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      loss += pyramid_levels_[i]->Forward(pooling_bottom_vecs_[i],
          &(pooling_top_vecs_[i]));
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
void SpatialPyramidPoolingLayer<Dtype>::Backward_cpu(
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
