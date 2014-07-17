// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using std::min;
using std::max;
using std::list;

namespace caffe {

class ScoredBoxes {
 public:
  ScoredBoxes(int y1, int x1, int y2, int x2, float score,
      int class_id, int box_id): y1_(y1), x1_(x1), y2_(y2), x2_(x2),
      score_(score), class_id_(class_id), box_id_(box_id) {
      area_ = (y2_ - y1_ + 1) * (x2_ - x1_ + 1);
  }
  float get_score() const { return score_; }
  int get_class_id() const { return class_id_; }
  int get_box_id() const { return box_id_; }
  float IoU (const ScoredBoxes& another_box) const {
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

inline bool operator>(const ScoredBoxes& box1, const ScoredBoxes& box2) {
  return box1.get_score() > box2.get_score();
}

void runNMS(list<ScoredBoxes>& sboxes_list, float nms_th);

template <typename Dtype>
void NMSLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[1]->width(), 4) << "proposal dimension must be 1*1*n*4";
  CHECK_GE(bottom[1]->height(), 0) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->channels(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->num(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->height(), bottom[0]->num()) << "proposal number must match the num of fc8";
  CHECK_EQ(bottom[0]->channels(), bottom[2]->count()) << "class number of fc8 must be consistent with class mask";
  proposal_num_ = bottom[1]->height();
  class_num_ = bottom[0]->channels();
  NMSParameter nms_param = this->layer_param_.nms_param();
  score_th_ = nms_param.score_th();
  nms_th_1_ = nms_param.nms_th_1();
  nms_th_2_ = nms_param.nms_th_2();
  disp_num_ = nms_param.disp_num();
  // Binary vector of each proposal
  (*top)[0]->Reshape(1, 1, 3, proposal_num_);
}

template <typename Dtype>
Dtype NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // clear previous results
  nms_list_1_.clear();
  nms_list_2_.clear();
  Dtype* result_vecs = (*top)[0]->mutable_cpu_data();
  memset(result_vecs, 0, proposal_num_*sizeof(Dtype));
  const Dtype* score_mat = bottom[0]->cpu_data();
  const Dtype* window_proposals = bottom[1]->cpu_data();
  const Dtype* class_mask = bottom[2]->cpu_data();
  // Within-class NMS
  for (int class_id = 0; class_id < class_num_; class_id++) {
    if (!class_mask[class_id])
      continue;
    // Do NMS for this class
    for (int box_id = 0; box_id < proposal_num_; box_id++) {
      float score = score_mat[class_num_ * box_id + class_id];
      if (score > score_th_) {
        int y1 = window_proposals[4*box_id];
        int x1 = window_proposals[4*box_id+1];
        int y2 = window_proposals[4*box_id+2];
        int x2 = window_proposals[4*box_id+3];
        // [0, 0, 0, 0] marks the end of valid boxes
        if (!y1 && !x1 && !y2 && !x2) {
          break;
        }
        nms_list_1_.push_back(ScoredBoxes(y1, x1, y2, x2, score, class_id, box_id));
        runNMS(nms_list_1_, nms_th_1_);
      }
    }
    nms_list_2_.insert(nms_list_2_.end(), nms_list_1_.begin(), nms_list_1_.end());
  }
  // Between-class NMS
  runNMS(nms_list_2_, nms_th_2_);
  // Write results
  int count = 0;
  list<ScoredBoxes>::iterator iter;
  for (iter = nms_list_2_.begin();
      (iter != nms_list_2_.end()) && (count < disp_num_); ++iter, ++count) {
    int box_id = int(iter->get_box_id());
    result_vecs[box_id] = 1;
    result_vecs[box_id + proposal_num_] = iter->get_class_id();
    result_vecs[box_id + 2*proposal_num_] = iter->get_score();
  }
  return Dtype(0.);
}

void runNMS(list<ScoredBoxes>& sboxes_list, float nms_th) {
  sboxes_list.sort(operator>); // sort the boxes into descending score
  list<ScoredBoxes>::iterator iter1, iter2, highest_box, to_remove, critical;
  for (iter1 = sboxes_list.begin(); iter1 != sboxes_list.end(); ) {
    highest_box = iter1++;
    critical = iter1;
    ++critical;
    for (iter2 = iter1; iter2 != sboxes_list.end(); ) {
      to_remove = iter2++;
      float IoU = highest_box->IoU(*to_remove);
      if (IoU > nms_th) {
        if (iter1 == to_remove) {
          ++iter1;
          sboxes_list.erase(to_remove);
        } else if (critical == to_remove) {
          sboxes_list.erase(to_remove);
          iter1 = iter2;
          --iter1;
        } else {
          sboxes_list.erase(to_remove);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(NMSLayer);

}  // namespace caffe
