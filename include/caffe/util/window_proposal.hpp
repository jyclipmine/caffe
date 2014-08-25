#ifndef CAFFE_UTIL_WINDOW_PROPOSAL_H_
#define CAFFE_UTIL_WINDOW_PROPOSAL_H_

#include <opencv2/opencv.hpp>

namespace caffe {

int window_proposal_bing(const cv::Mat& img, float boxes[],
    const int max_proposal_num);
int window_proposal_fixed(const cv::Mat& img, float boxes[],
    const int max_proposal_num);

}  // namespace caffe

#endif // CAFFE_UTIL_WINDOW_PROPOSAL_H_
