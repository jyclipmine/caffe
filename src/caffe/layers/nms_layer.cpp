// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::min;
using std::max;

namespace caffe {

template <typename Dtype>
void NMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Check parameters
  CHECK_EQ(bottom[0]->width(), 1) << "fc top blob must have width == 1";
  CHECK_EQ(bottom[0]->height(), 1) << "fc top blob must have height == 1";
  CHECK_EQ(bottom[1]->width(), 4) << "proposal dimension must be 1*1*n*4";
  CHECK_GE(bottom[1]->height(), 0) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->channels(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->num(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->height(), bottom[0]->num())
      << "proposal number must match the num of fc8";
  CHECK_EQ(bottom[2]->count(), bottom[0]->num())
      << "the length of input valid vector must match the num of fc8";
  proposal_num_ = bottom[1]->height();
  class_num_ = bottom[0]->channels();
  NMSParameter nms_param = this->layer_param_.nms_param();
  score_th_ = nms_param.score_th();
  within_class_nms_th_ = nms_param.within_class_nms_th();
  across_class_nms_th_ = nms_param.across_class_nms_th(); 
  // set up cache
  blob_keep_vec_per_class_.reset(new Blob<Dtype>(bottom[0]->num(), 
      bottom[0]->channels(), bottom[0]->height(), bottom[0]->width())); 
  // set up output blobs
  (*top)[0]->Reshape(proposal_num_, 1, 1, 1);
  (*top)[1]->Reshape(proposal_num_, 1, 1, 1);
  (*top)[2]->Reshape(proposal_num_, 1, 1, 1);
}

template <typename Dtype>
void NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(NMSLayer);

}  // namespace caffe
