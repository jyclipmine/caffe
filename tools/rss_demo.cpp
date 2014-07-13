#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp> 

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

// int runNet(int argc, char** argv);

// get a 688 by 688 demo
const Mat& read_from_camera(CvCapture* pCapture) {
  IplImage* pFrame = cvQueryFrame(pCapture);
  waitKey(20);
  static Mat camera_input(pFrame, 0);
  return camera_input;
}

// void RunBING(Mat& img);
int main() {
	CvCapture* pCapture = cvCreateCameraCapture(0);
  while (true) {
    imshow(read_from_camera(pCapture));
  }
  return 0;
}
