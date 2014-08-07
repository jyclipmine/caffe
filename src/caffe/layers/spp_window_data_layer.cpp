// Copyright 2014 BVLC and contributors.
//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>
#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::map;
using std::pair;
using std::ifstream;

namespace caffe {

template <typename Dtype>
void* SPPWindowDataLayerPrefetch(void* layer_pointer) {
  SPPWindowDataLayer<Dtype>* layer =
      reinterpret_cast<SPPWindowDataLayer<Dtype>*>(layer_pointer);

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  // zero out batch
  caffe_set(layer->prefetch_data_->count(), Dtype(0), top_data);

  const int batch_size =
      layer->layer_param_.spp_window_data_param().batch_size();
  const int batch_per_file =
      layer->layer_param_.spp_window_data_param().batch_per_file();
  const int feat_dim = layer->layer_param_.spp_window_data_param().feat_dim();
  ifstream& feat_ifs = layer->feat_ifs;

  // open new feature cache file 
  if (layer->open_new_file_) {
    char id_str[16];
    snprintf(id_str, sizeof(id_str), "%d", layer->current_file_id_);
    string file_path = layer->cache_dir_ + "/" + id_str + layer->extension_;
    feat_ifs.open(file_path.c_str(), std::ios::binary);
    CHECK(feat_ifs.good()) << "Failed to open feature file " << file_path;
    // check file parameteres
    float feat_file_param[3];
    feat_ifs.read(reinterpret_cast<char*>(feat_file_param),
        sizeof(feat_file_param));
    CHECK_EQ(feat_file_param[1], batch_size) << "batch size mismatch";
    CHECK_EQ(feat_file_param[2], feat_dim) << "feature dimension mismatch";
    layer->actual_batch_num_ = feat_file_param[0];
    layer->current_batch_id_ = 0;
    if (layer->actual_batch_num_  != batch_per_file) {
      LOG(INFO) << "there are " << layer->actual_batch_num_
          << "batches in file " << file_path;
    }
  }
  // read label
  feat_ifs.read(reinterpret_cast<char*>(top_label), sizeof(Dtype) * batch_size);
  // read feature data
  feat_ifs.read(reinterpret_cast<char*>(top_data),
      sizeof(Dtype) * feat_dim * batch_size);
  // check if stream is good
  CHECK(feat_ifs.good()) << "Error occurred while reading batch id = "
      << layer->current_batch_id_ << " of " << layer->actual_batch_num_ 
      << " batches";
  // move to next batch
  layer->current_batch_id_++;
  // check if this is the last batch in the file
  if (layer->current_batch_id_ >= layer->actual_batch_num_) {
    feat_ifs.close();
    layer->current_batch_id_ = 0;
    layer->actual_batch_num_ = 0;
    // move to next file
    layer->current_file_id_++;
    // check if this is the last file
    if (layer->current_file_id_ >= layer->file_num_) {
      layer->current_file_id_ = 0;
    }
    layer->open_new_file_ = true;
  }
  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
SPPWindowDataLayer<Dtype>::~SPPWindowDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void SPPWindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2
  LOG(INFO) << "SPPWindowData layer:"
      << "\n  feature cache directory: "
      << this->layer_param_.spp_window_data_param().cache_dir()
      << "\n  extension: "
      << this->layer_param_.spp_window_data_param().extension()
      << "\n  batch size: "
      << this->layer_param_.spp_window_data_param().batch_size()
      << "\n  batch per file: "
      << this->layer_param_.spp_window_data_param().batch_per_file()
      << "\n  file num: "
      << this->layer_param_.spp_window_data_param().file_num()
      << "\n  feature dimension: "
      << this->layer_param_.spp_window_data_param().feat_dim() << std::endl;
      
  cache_dir_ = this->layer_param_.spp_window_data_param().cache_dir();
  extension_ = this->layer_param_.spp_window_data_param().extension();
  file_num_ = this->layer_param_.spp_window_data_param().file_num();
  current_file_id_ = 0;
  actual_batch_num_ = 0;
  current_batch_id_ = 0;
  open_new_file_ = true;

  // construct data blob and label blob
  const int batch_size =
      this->layer_param_.spp_window_data_param().batch_size();
  const int feat_dim = this->layer_param_.spp_window_data_param().feat_dim();
  (*top)[0]->Reshape(batch_size, 1, 1, feat_dim);
  prefetch_data_.reset(new Blob<Dtype>(batch_size, 1, 1, feat_dim));
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  prefetch_label_.reset(new Blob<Dtype>(batch_size, 1, 1, 1));

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void SPPWindowDataLayer<Dtype>::CreatePrefetchThread() {
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, SPPWindowDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void SPPWindowDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int SPPWindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype SPPWindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(SPPWindowDataLayer);

}  // namespace caffe
