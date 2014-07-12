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
using std::min;
using std::max

namespace caffe {

void runNMS(list<ScoredBoxes>& sboxes_list, float nms_th);

template <typename Dtype>
void SPPDetectorLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[1]->width(), 4) << "proposal dimension must be 1*1*n*4";
  CHECK_GE(bottom[1]->height(), 0) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->channels(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->num(), 1) << "proposal dimension must be 1*1*n*4";
  CHECK_EQ(bottom[1]->height(), bottom[0]->num()) << "proposal number must match the num of fc8";
  CHECK_EQ(bottom[0]->channels(), bottom[2]->count()) << "class number of fc8 must be consistent with class mask";
  proposal_num_ = bottom[1]->height();
  class_num_ = height[0]->channels();
  NMSParameter nms_param = this->layer_param_.nms_param();
  score_th_ = nms_param.score_th();
  nms_th_1_ = nms_param.nms_th_1();
  nms_th_2_ = nms_param.nms_th_2();
  disp_num_ = nms_param.disp_num();
  // Binary vector of each proposal
  (*top)[0]->Reshape(proposal_num_, 1, 1, 1);
  // Class id of each proposal
  (*top)[1]->Reshape(proposal_num_, 1, 1, 1);
}

template <typename Dtype>
Dtype SPPDetectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // clear previous results
  nms_list_1_.clear();
  nms_list_2_.clear();
  Dtype* keep_vec = (*top[0])->cpu_data();
  Dtype* class_vec = (*top[1])->cpu_data();
  memset(keep_vec, 0, proposal_num_*sizeof(Dtype));
  memset(class_vec, 0, proposal_num_*sizeof(Dtype));
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
        nms_list_1_.push_back(ScoredBoxes(y1, x1, y2, x2, score, class_id, box_id));
        runNMS(nms_list_1_, nms_th_1);
      }
    }
    nms_list_2_.insert(nms_list_2_.end(), nms_list_1_.begin(), nms_list_1_.end());
  }
  // Between-class NMS
  runNMS(nms_list_1_, nms_th_1);
  // Write results
  for (list<ScoredBoxes>::iterator iter = nms_list_2_.begin();
      iter != nms_list_2_.end(); ++iter) {
    int box_id = int(iter->get_box_id());
    keep_vec[box_id] = 1;
    class_vec[box_id] = iter->get_class_id();
  }
  return Dtype(0.);
}

void runNMS(list<ScoredBoxes>& sboxes_list, float nms_th) {
  sboxes_list.sort(operator>); // sort the boxes into descending score
  list<ScoredBoxes>::iterator iter1, iter2, highest, to_remove;
  for (iter1 = sboxes_list.begin(); iter1 != sboxes_list.end(); ) {
    highest_box = iter1++;
    for (iter2 = iter1; iter2 != sboxes_list.end(); ) {
      to_remove = iter2++;
      float IoU = highest_box->IoU(to_remove);
      if (IoU > nms_th) {
        if (iter1 == to_remove) {
          ++iter1;
          sboxes_list.erase(to_remove);
        } else if (advance(iter1, 1) == to_remove)
          sboxes_list.erase(to_remove);
          iter1 = advance(iter2, -1)
        } else {
          sboxes_list.erase(to_remove);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SPPDetectorLayer);

}  // namespace caffe
