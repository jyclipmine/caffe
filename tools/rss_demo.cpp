#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

int runBING(Mat& image, float boxes[], float conv5_windows[], const int boxes_num,
    const int max_size, const int min_size,
    const int conv5_hend, const int conv5_wend);

// get a 640 by 480 demo
const IplImage* read_from_camera(CvCapture* pCapture) {
  IplImage* pFrame = cvQueryFrame(pCapture);
  waitKey(20);
  return pFrame;
}

// do mean subtraction and convert into float type
// Mat is already in BGR channel
void Mat2float(float image_data[], const Mat& image, const float channel_mean[]) {
  const int channels = 3;
  const int width = image.cols;
  const int height = image.rows;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        float pixel = static_cast<float>(image.at<cv::Vec3b>(h, w)[c]);
        image_data[w + width * (h + height * c)] = pixel - channel_mean[c];
      }
    }
  }
}

void load_channel_mean(float channel_mean[]) {
  const char* filename = "channel_mean.dat";
  ifstream fin(filename, ios::binary);
  CV_Assert(fin);
  fin.read(reinterpret_cast<char*>(channel_mean), 3*sizeof(float));
  CV_Assert(fin);
  fin.close();
  cout << "Channel mean: B = " << channel_mean[0] << ", G = " << channel_mean[1]
      << ", R = " << channel_mean[2] << endl;
}

void load_class_mask(float class_mask[]) {
  for (int i = 0; i < 7405; i++)
    class_mask[i] = 1;
}

const float* forward_network(Net<float>& net, float image_data[], float conv5_windows[],
    float boxes[], float class_mask[]) {
  vector<Blob<float>*>& output_blobs = net.output_blobs();
  vector<Blob<float>*>& input_blobs = net.input_blobs();
  memcpy(input_blobs[0]->mutable_cpu_data(), image_data,
      sizeof(float) * input_blobs[0]->count());  
  memcpy(input_blobs[1]->mutable_cpu_data(), conv5_windows,
      sizeof(float) * input_blobs[1]->count());       
  memcpy(input_blobs[2]->mutable_cpu_data(), boxes,
      sizeof(float) * input_blobs[2]->count());       
  memcpy(input_blobs[3]->mutable_cpu_data(), class_mask,
      sizeof(float) * input_blobs[3]->count());
  const vector<Blob<float>*>& result = net.ForwardPrefilled();
  return result[0]->cpu_data();
}

void draw_results(Mat& image, const float result_vecs[], float boxes[],
    int proposal_num) {
  const static CvScalar color = cvScalar(0, 0, 255);
  const float* keep_vec = result_vecs;
  const float* class_id_vec = result_vecs + proposal_num;
  const float* score_vec = result_vecs + proposal_num*2;
  for (int box_id = 0; box_id < proposal_num; box_id++) {
    if (keep_vec[box_id]) {
      int y1 = boxes[box_id*4  ];
      int x1 = boxes[box_id*4+1];
      int y2 = boxes[box_id*4+2];
      int x2 = boxes[box_id*4+3];
      Point ul(x1, y1), ur(x2, y1), ll(x1, y2), lr(x2, y2);
      line(image, ul, ur, color, 3);
      line(image, ur, lr, color, 3);
      line(image, lr, ll, color, 3);
      line(image, ll, ul, color, 3);
    }
  }
}

int main(int argc, char** argv) {
  CV_Assert(argc == 3);
  
  // Parameters
	CvCapture* pCapture = cvCreateCameraCapture(0);
	const int proposal_num = 2000;
	const int conv5_hend = 29, conv5_wend = 39;
	const int image_h = 480, image_w = 640;
	const int max_size = 350, min_size = 80;
	const int class_num = 7405;
	float boxes[proposal_num*4];
	float conv5_windows[proposal_num*4];
	float image_data[image_h*image_w*3];
	float channel_mean[3];
	float class_mask[class_num];
	load_channel_mean(channel_mean);
	load_class_mask(class_mask);
	
	// Initialize network
	Caffe::set_phase(Caffe::TEST);
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(1);
  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);


  // run loop
  while (true) {
    Mat image(read_from_camera(pCapture), true); // copy data
    CV_Assert((image.cols == image_w) && (image.rows == image_h));
    int count = runBING(image, boxes, conv5_windows, proposal_num,
        max_size, min_size, conv5_hend, conv5_wend);
    Mat2float(image_data, image, channel_mean);
    
    const float* result_vecs = forward_network(caffe_test_net, image_data,
        conv5_windows, boxes, class_mask);
    draw_results(image, result_vecs, boxes, proposal_num);
    imshow("detection results", image);
  }
  return 0;
}
