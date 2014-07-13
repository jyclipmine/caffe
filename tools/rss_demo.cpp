#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp> 

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

// int runNet(int argc, char** argv);

// get a 688 by 688 demo
void read_from_camera(Mat& img) {
  
}

// void RunBING(Mat& img);

int main() {
  
  //IplImage* detect_result_color = cvCreateImage(cvSize(FRAME_W, FRAME_H), 8, 3);
	//IplImage* detect_result_gray  = cvCreateImage(cvSize(FRAME_W, FRAME_H), 8, 1);
	IplImage* pFrame = 0;
	CvCapture* pCapture = cvCreateCameraCapture(0);
  while (true) {
  pFrame = cvQueryFrame(pCapture);
	cvShowImage("head-shoulder detection (color)", pFrame);
    // unsigned int width = getWidthOfPhoto();
    // unsinged int height = getHeightOfPhoto();
    // unsinged char* dataBuffer = getBufferOfPhoto();
    // Mat image(Size(width, height), CV_8UC1, dataBuffer, Mat::AUTO_STEP);
    // imshow("image", image);
    waitKey();
  }
}


