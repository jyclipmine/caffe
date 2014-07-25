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

namespace caffe {

template <typename Dtype>
void* SPPWindowDataLayerPrefetch(void* layer_pointer) {
  SPPWindowDataLayer<Dtype>* layer =
      reinterpret_cast<SPPWindowDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const int batch_size = layer->layer_param_.spp_window_data_param().batch_size();
  const int spp5_dim = layer->layer_param_.spp_window_data_param().spp5_dim();
  const float fg_fraction =
      layer->layer_param_.spp_window_data_param().fg_fraction();

  // zero out batch
  caffe_set(layer->prefetch_data_->count(), Dtype(0), top_data);
  // input file stream
  std::ifstream cache_ifs;
  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };
  char cache_name[128];
  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      const unsigned int rand_index = layer->PrefetchRand();
      vector<float> window = (is_fg) ?
          layer->fg_windows_[rand_index % layer->fg_windows_.size()] :
          layer->bg_windows_[rand_index % layer->bg_windows_.size()];

      pair<std::string, vector<int> > image =
          layer->image_database_[window[SPPWindowDataLayer<Dtype>::IMAGE_INDEX]];
          
      // get feature cache file path
      int image_index = window[SPPWindowDataLayer<Dtype>::IMAGE_INDEX];
      int x1 = window[SPPWindowDataLayer<Dtype>::X1];
      int y1 = window[SPPWindowDataLayer<Dtype>::Y1];
      int x2 = window[SPPWindowDataLayer<Dtype>::X2];
      int y2 = window[SPPWindowDataLayer<Dtype>::Y2];
      sprintf(cache_name, "%d/%d_%d_%d_%d.spp5feat", image_index, x1, y1, x2, y2);
      string cache_file_path = layer->cache_dir_ + "/" + cache_name;
      
      // read feature into cpu memory
      Dtype* target_addr = top_data + spp5_dim * item_id;
      int read_bytes = sizeof(Dtype) * spp5_dim;
      cache_ifs.open(cache_file_path.c_str(), std::ios::binary);
      CHECK(cache_ifs.good()) << "Failed to open spp5 feature file "
          << cache_file_path << ", which corresponds to image "
          << image.first;
      cache_ifs.read(reinterpret_cast<char*>(target_addr), read_bytes);
      CHECK(cache_ifs.eof()) << "Error loading to open spp5 feature file "
          << cache_file_path << ", which corresponds to image "
          << image.first;
      cache_ifs.close();

      // get window label
      top_label[item_id] = window[SPPWindowDataLayer<Dtype>::LABEL];
      item_id++;
    }
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

  LOG(INFO) << "SPPWindowData layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.spp_window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.spp_window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.spp_window_data_param().fg_fraction() << std::endl
      << "  spp5 feature dimension: "
      << this->layer_param_.spp_window_data_param().spp5_dim() << std::endl
      << "  spp5 feature cache directory: "
      << this->layer_param_.spp_window_data_param().cache_dir() << std::endl;
  cache_dir_ = this->layer_param_.spp_window_data_param().cache_dir();
  
  std::ifstream infile(this->layer_param_.spp_window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.spp_window_data_param().source() << std::endl;
  
  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    image_database_.push_back(std::make_pair(image_path, image_size));

    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.spp_window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.spp_window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;

      vector<float> window(SPPWindowDataLayer::NUM);
      window[SPPWindowDataLayer::IMAGE_INDEX] = image_index;
      window[SPPWindowDataLayer::LABEL] = label;
      window[SPPWindowDataLayer::OVERLAP] = overlap;
      window[SPPWindowDataLayer::X1] = x1;
      window[SPPWindowDataLayer::Y1] = y1;
      window[SPPWindowDataLayer::X2] = x2;
      window[SPPWindowDataLayer::Y2] = y2;

      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        int label = window[SPPWindowDataLayer::LABEL];
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[SPPWindowDataLayer::LABEL] = 0;
        window[SPPWindowDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
      }
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  const int batch_size = this->layer_param_.spp_window_data_param().batch_size();
  const int spp5_dim = this->layer_param_.spp_window_data_param().spp5_dim();
  (*top)[0]->Reshape(batch_size, 1, 1, spp5_dim);
  prefetch_data_.reset(new Blob<Dtype>(batch_size, 1, 1, spp5_dim));

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
  const bool prefetch_needs_rand =
      this->layer_param_.spp_window_data_param().crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
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
