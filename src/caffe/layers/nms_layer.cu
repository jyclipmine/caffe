// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
__device__ Dtype deviceIoU(Dtype b1_y1, Dtype b1_x1, Dtype b1_y2, Dtype b1_x2,
    Dtype b2_y1, Dtype b2_x1, Dtype b2_y2, Dtype b2_x2) {
  Dtype b1_area = (b1_y2 - b1_y1 + 1) * (b1_x2 - b1_x1 + 1);
  Dtype b2_area = (b2_y2 - b2_y1 + 1) * (b2_x2 - b2_x1 + 1);
  Dtype max_y1 = max(b1_y1, b2_y1);
  Dtype max_x1 = max(b1_x1, b2_x1);
  Dtype min_y2 = min(b1_y2, b2_y2);
  Dtype min_x2 = min(b1_x2, b2_x2);
  Dtype inter_area = max(min_y2 - max_y1 + 1, 0.f)
      * max(min_x2 - max_x1 + 1, 0.f);
  Dtype IoU = inter_area / (b1_area + b2_area - inter_area);
  return IoU;
}

template <typename Dtype>
__global__ void kernel_within_class_nms(const Dtype* bottom_data,
    const Dtype* boxes, const Dtype* valid_vec, const int proposal_num,
    const int class_num, const Dtype score_th, const Dtype within_class_nms_th,
    Dtype* keep_vec_per_class) {
  CUDA_KERNEL_LOOP(index, proposal_num * class_num) {
    int n_this = index / class_num;
    int cls = index % class_num;
    int index_this = n_this * class_num + cls;
    if (valid_vec[n_this]) {
      // use float-point coordinates for boxes
      Dtype y1_this = boxes[4*n_this];
      Dtype x1_this = boxes[4*n_this+1];
      Dtype y2_this = boxes[4*n_this+2];
      Dtype x2_this = boxes[4*n_this+3];
      Dtype score_this = bottom_data[index_this];
      // suppress this box if its score is lower than score_th
      if (score_this < score_th) {
        keep_vec_per_class[index_this] = false;
      }
      // compare the box with all other boxes in this class
      for (int n_comp = 0; n_comp < proposal_num; n_comp++) {
        if (n_comp == n_this)
          continue;
        int index_comp = n_comp * class_num + cls;
        if (valid_vec[n_comp]) {
          Dtype y1_comp = boxes[4*n_comp];
          Dtype x1_comp = boxes[4*n_comp+1];
          Dtype y2_comp = boxes[4*n_comp+2];
          Dtype x2_comp = boxes[4*n_comp+3];
          Dtype score_comp = bottom_data[index_comp];
          // suppress comp box its score is lower than score_th
          if (score_comp < score_th) {
            keep_vec_per_class[index_comp] = false;
          }
          // if IoU is higher than threshold, suppress the box with lower scores
          Dtype IoU = deviceIoU(y1_this, x1_this, y2_this, x2_this,
              y1_comp, x1_comp, y2_comp, x2_comp);
          if (IoU > within_class_nms_th) {
            int index_remove =
                (score_comp < score_this ? index_comp : index_this);
            keep_vec_per_class[index_remove] = false;
          }
        } else {
          keep_vec_per_class[index_comp] = false;
        }
      }
    } else {
      keep_vec_per_class[index_this] = false;
    }
  }
}

template <typename Dtype>
__global__ void kernel_find_class_id(const Dtype* bottom_data, 
    const Dtype* keep_vec_per_class, const int proposal_num,
    const int class_num, Dtype* keep_vec, Dtype* class_id_vec,
    Dtype* score_vec) {
  CUDA_KERNEL_LOOP(n, proposal_num) {
    bool keep_this = false;
    Dtype max_score = -FLT_MAX;
    int max_cls = -1;
    for (int cls = 0; cls < class_num; cls++) {
      if (keep_vec_per_class[n * class_num + cls]) {
        keep_this = true;
        if (bottom_data[n * class_num + cls] > max_score) {
          max_cls = cls;
          max_score = bottom_data[n * class_num + cls];
        }
      }
    }
    keep_vec[n] = keep_this;
    class_id_vec[n] = max_cls;
    score_vec[n] = max_score;
  }
}

template <typename Dtype>
__global__ void kernel_across_class_nms(const Dtype* boxes,
    const Dtype* score_vec, const int proposal_num, const int class_num,
    const Dtype across_class_nms_th, Dtype* keep_vec) {
  CUDA_KERNEL_LOOP(n_this, proposal_num) {
    if (keep_vec[n_this]) {
      // use float-point coordinates for boxes
      Dtype y1_this = boxes[4*n_this];
      Dtype x1_this = boxes[4*n_this+1];
      Dtype y2_this = boxes[4*n_this+2];
      Dtype x2_this = boxes[4*n_this+3];
      Dtype score_this = score_vec[n_this];
      // compare the box with all other boxes in this class
      for (int n_comp = 0; n_comp < proposal_num; n_comp++) {
        if (n_comp == n_this)
          continue;
        if (keep_vec[n_comp]) {
          Dtype y1_comp = boxes[4*n_comp];
          Dtype x1_comp = boxes[4*n_comp+1];
          Dtype y2_comp = boxes[4*n_comp+2];
          Dtype x2_comp = boxes[4*n_comp+3];
          Dtype score_comp = score_vec[n_comp];
          // if IoU is higher than threshold, suppress the box with lower scores
          Dtype IoU = deviceIoU(y1_this, x1_this, y2_this, x2_this,
              y1_comp, x1_comp, y2_comp, x2_comp);
          if (IoU > across_class_nms_th) {
            int n_remove = (score_comp < score_this ? n_comp : n_this);
            keep_vec[n_remove] = false;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void NMSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* boxes = bottom[1]->gpu_data();
  const Dtype* valid_vec = bottom[2]->gpu_data();
  Dtype* keep_vec_per_class = blob_keep_vec_per_class_->mutable_gpu_data();
  Dtype* keep_vec = (*top)[0]->mutable_gpu_data();
  Dtype* class_id_vec = keep_vec + proposal_num_;
  Dtype* score_vec = keep_vec + proposal_num_ * 2;
  // reset keep_vec_per_class to be all true
  caffe_gpu_set(proposal_num_ * class_num_, Dtype(true), keep_vec_per_class);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_within_class_nms<<<CAFFE_GET_BLOCKS(proposal_num_ * class_num_),
      CAFFE_CUDA_NUM_THREADS>>>(bottom_data, boxes, valid_vec, proposal_num_,
      class_num_, score_th_, within_class_nms_th_, keep_vec_per_class);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_find_class_id<<<CAFFE_GET_BLOCKS(proposal_num_),
      CAFFE_CUDA_NUM_THREADS>>>(bottom_data, keep_vec_per_class, proposal_num_,
      class_num_, keep_vec, class_id_vec, score_vec);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_across_class_nms<<<CAFFE_GET_BLOCKS(proposal_num_),
      CAFFE_CUDA_NUM_THREADS>>>(boxes, score_vec, proposal_num_, class_num_,
      across_class_nms_th_, keep_vec);
}

INSTANTIATE_CLASS(NMSLayer);

}  // namespace caffe
