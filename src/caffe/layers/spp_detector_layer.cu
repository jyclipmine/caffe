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
  if (bottom[0]->gpu_data() != (*top)[0]->gpu_data()) {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
             (*top)[0]->mutable_gpu_data());
  }
  if (bottom[1]->gpu_data() != (*top)[1]->gpu_data()) {
    caffe_copy(bottom[1]->count(), bottom[1]->gpu_data(),
             (*top)[1]->mutable_gpu_data());
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
