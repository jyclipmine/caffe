#ifndef CAFFE_UTIL_WINDOW_PROPOSAL_H_
#define CAFFE_UTIL_WINDOW_PROPOSAL_H_

#include <opencv2/opencv.hpp>

namespace caffe {

class WindowProposer {
public:
  virtual int propose(const cv::Mat& img, float boxes[],
      const int max_proposal_num) = 0;
};

class BINGWindowProposer: public WindowProposer {
public:
  virtual int propose(const cv::Mat& img, float boxes[],
      const int max_proposal_num);
};

class FixedWindowProposer: public WindowProposer {
public:
  virtual int propose(const cv::Mat& img, float boxes[],
      const int max_proposal_num);
};

class GOPWindowProposer: public WindowProposer {
public:
  GOPWindowProposer();
  ~GOPWindowProposer();
  virtual int propose(const cv::Mat& img, float boxes[],
      const int max_proposal_num);
private:
  void* parameters_;
};

}  // namespace caffe

#endif // CAFFE_UTIL_WINDOW_PROPOSAL_H_
