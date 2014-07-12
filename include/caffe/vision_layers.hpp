// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <list>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::list;

namespace caffe {

/* ConvolutionLayer
*/
template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_CONVOLUTION;
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int kernel_size_;
  int stride_;
  int num_;
  int channels_;
  int pad_;
  int height_;
  int width_;
  int num_output_;
  int group_;
  Blob<Dtype> col_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool bias_term_;
  int M_;
  int K_;
  int N_;
};

/* EltwiseLayer
  Compute elementwise operations like product or sum.
*/
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
 public:
  explicit EltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_ELTWISE;
  }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
};

/* Im2colLayer
*/
template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_IM2COL;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int kernel_size_;
  int stride_;
  int channels_;
  int height_;
  int width_;
  int pad_;
};

/* InnerProductLayer
*/
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_INNER_PRODUCT;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  shared_ptr<SyncedMemory> bias_multiplier_;
};

// Forward declare PoolingLayer and SplitLayer for use in LRNLayer.
template <typename Dtype> class PoolingLayer;
template <typename Dtype> class SplitLayer;

/* LRNLayer
  Local Response Normalization
*/
template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LRN;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  virtual Dtype CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  int num_;
  int channels_;
  int height_;
  int width_;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_input_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_bottom_vec_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<PoolingLayer<Dtype> > pool_layer_;
  Blob<Dtype> pool_output_;
  vector<Blob<Dtype>*> pool_top_vec_;
  shared_ptr<PowerLayer<Dtype> > power_layer_;
  Blob<Dtype> power_output_;
  vector<Blob<Dtype>*> power_top_vec_;
  shared_ptr<EltwiseLayer<Dtype> > product_layer_;
  Blob<Dtype> product_data_input_;
  vector<Blob<Dtype>*> product_bottom_vec_;
};

/* PoolingLayer
*/
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_POOLING;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return max_top_blobs_; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int max_top_blobs_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Blob<Dtype> rand_idx_;
  shared_ptr<Blob<int> > max_idx_;
};

/* PyramidLevelLayer
*/
template <typename Dtype>
class PyramidLevelLayer : public Layer<Dtype> {
 public:
  explicit PyramidLevelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_POOLING;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return max_top_blobs_; }
  void setROI(int roi_start_h, int roi_start_w, int roi_end_h, int roi_end_w);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int max_top_blobs_;
  int bin_num_h_;
  int bin_num_w_;
  float bin_size_h_;
  float bin_size_w_;
  int channels_;
  int height_;
  int width_;
  int roi_start_h_;
  int roi_start_w_;
  int roi_end_h_;
  int roi_end_w_;
  Blob<Dtype> rand_idx_;
  shared_ptr<Blob<int> > max_idx_;
};

/* SpatialPyramidPoolingLayer
*/
template <typename Dtype>
class SpatialPyramidPoolingLayer : public Layer<Dtype> {
 public:
  explicit SpatialPyramidPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SPATIAL_PYRAMID_POOLING;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  void setROI(int roi_start_h, int roi_start_w, int roi_end_h, int roi_end_w);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int num_pyramid_levels_;
  int channels_;
  int height_;
  int width_;
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  vector<shared_ptr<PyramidLevelLayer<Dtype> > > pyramid_levels_;
  vector<shared_ptr<FlattenLayer<Dtype> > > flatten_layers_;
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
  vector<vector<Blob<Dtype>*> > pooling_bottom_vecs_;
  vector<vector<Blob<Dtype>*> > pooling_top_vecs_;
  vector<vector<Blob<Dtype>*> > flatten_top_vecs_;
  vector<Blob<Dtype>*> concat_bottom_vec_;
};

/* SPPDetectorLayer
*/
template <typename Dtype>
class SPPDetectorLayer : public Layer<Dtype> {
 public:
  explicit SPPDetectorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SPP_DETECTOR;
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    return Forward_cpu(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    LOG(FATAL) << "Backward not supported for SPPDetectorLayer";
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    LOG(FATAL) << "Backward not supported for SPPDetectorLayer";
  }

  int proposal_num_;
  int dim_;
  vector<shared_ptr<SpatialPyramidPoolingLayer<Dtype> > > spp_layers_;
  vector<vector<Blob<Dtype>*> > spp_bottom_vecs_;
  vector<vector<Blob<Dtype>*> > spp_top_vecs_;
};

class ScoredBoxes {
 public:
  ScoredBoxes(int y1, int x1, int y2, int x2, float score,
      int class_id, int box_id): y1_(y1), x1_(x1), y2_(y2), x2_(x2),
      score_(score), class_id_(class_id), box_id_(box_id) {
      area_ = (y2_ - y1_ + 1) * (x2_ - x1_ + 1);
  }
  float get_score() { return score_; }
  int get_class_id() { return class_id_; }
  int get_box_id() { return box_id_; }
  float IoU(const ScoredBoxes& another_box) {
    int yy1 = std::max(this->y1_, another_box.y1_);
    int xx1 = std::max(this->x1_, another_box.x1_);
    int yy2 = std::min(this->y2_, another_box.y2_);
    int xx2 = std::min(this->x2_, another_box.x2_);
    int inter = std::max(0, yy2 - yy1 + 1) * std::max(0, xx2 - xx1 + 1);
    return float(inter) / float(this->area_ + another_box.area_ - inter);
  }
  
 private:
  int y1_;
  int x1_;
  int y2_;
  int x2_;
  float score_;
  int class_id_;
  int box_id_;
  int area_;
};

inline bool operator>(const ScoredBoxes& box1, const ScoredBoxes& box2)  {
  return box1.get_score() > box2.get_score();
}

/* NMSLayer
*/
template <typename Dtype>
class NMSLayer : public Layer<Dtype> {
 public:
  explicit NMSLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_NMS;
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    return Forward_cpu(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    LOG(FATAL) << "Backward not supported for NMSLayer";
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    LOG(FATAL) << "Backward not supported for NMSLayer";
  }

  float score_th_;
  float nms_th_1_;
  float nms_th_2_;
  int disp_num_;
  int proposal_num_;
  int class_num_;
  list<ScoredBoxes> nms_list_1_;
  list<ScoredBoxes> nms_list_2_;
};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
