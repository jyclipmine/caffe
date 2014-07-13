#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp> 

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

int runNet(int argc, char** argv);

// get a 688 by 688 demo
void read_from_camera(Mat& img) {
  
}

void RunBING(Mat& img);

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

int runNet(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU] [Device ID]";
    return 1;
  }

  Caffe::set_phase(Caffe::TEST);

  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);

  int total_iter = atoi(argv[3]);
  LOG(ERROR) << "Running " << total_iter << " iterations.";

  double test_accuracy = 0;
  for (int i = 0; i < total_iter; ++i) {
    const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
    test_accuracy += result[0]->cpu_data()[0];
    LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
  }
  test_accuracy /= total_iter;
  LOG(ERROR) << "Test accuracy: " << test_accuracy;

  return 0;
}
