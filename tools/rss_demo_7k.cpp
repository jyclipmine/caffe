#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <string>

using namespace caffe; // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

int bing_boxes(const Mat& img, float boxes[], const int max_proposal_num);
void boxes2conv5(const float boxes[], const int max_proposal_num,
    const int proposal_num, float conv5_windows[], float conv5_scales[]);
const IplImage* read_from_camera(CvCapture* pCapture);
void Mat2float(float image_data[], const Mat& img, const float channel_mean[]);
void load_channel_mean(float channel_mean[], const char* filename);
void load_class_name(vector<string>& class_name_vec, const char* filename);
void load_class_mask(float class_mask[]);
void draw_results(Mat& img, const float result_vecs[], float boxes[],
    int max_proposal_num, vector<string>& class_name_vec);
const float* forward_network(Net<float>& net, float image_data[],
    float conv5_windows[], float conv5_scales[], float boxes[],
    float class_mask[], const int class_num, const int max_proposal_num,
    const Mat& img);

void boxes2conv5(const float boxes[], const int max_proposal_num,
    const int proposal_num, float conv5_windows[], float conv5_scales[]) {
  const int conv5_stride = 16;
  // calculate the corresponing windows on conv5 feature map
  // zero out the boxes
  memset(conv5_windows, 0, max_proposal_num * 4 * sizeof(float));
  for (int i = 0; i < proposal_num; i++) {
    float y1 = boxes[4*i];
    float x1 = boxes[4*i+1];
    float y2 = boxes[4*i+2];
    float x2 = boxes[4*i+3];
    // round and add 1 to ends
    conv5_windows[4*i  ] = static_cast<int>(0.5f + y1 / conv5_stride);
    conv5_windows[4*i+1] = static_cast<int>(0.5f + x1 / conv5_stride);
    conv5_windows[4*i+2] = static_cast<int>(0.5f + y2 / conv5_stride) + 1;
    conv5_windows[4*i+3] = static_cast<int>(0.5f + x2 / conv5_stride) + 1;
  }
  // for now, set all scales to be zero
  memset(conv5_scales, 0, max_proposal_num * sizeof(float));
}

int main(int argc, char** argv) {
  CHECK_EQ(argc, 5) << "Input argument number mismatch";
  
  // Parameters
  CvCapture* pCapture = cvCreateCameraCapture(0);
  const int max_proposal_num = 1000;
  const int class_num = 7405;
  const int image_h = 688, image_w = 917;
  const int device_id = 1;
  const Size input_size(image_w, image_h);
  
  // Storage
  float boxes[max_proposal_num*4];
  float conv5_windows[max_proposal_num*4];
  float conv5_scales[max_proposal_num*4];
  float image_data[image_h*image_w*3];
  float channel_mean[3];
  float class_mask[class_num];
  
  // Initialize network
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(device_id);
  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  vector<string> class_name_vec(class_num);
  
  // Load data from disk
  load_channel_mean(channel_mean, argv[3]);
  load_class_name(class_name_vec, argv[4]);
  load_class_mask(class_mask);
  
  // timing
  clock_t start, finish;
  clock_t start_all, finish_all;
  // run loop
  while (true) {
    LOG(INFO) << "-------------------------------------------";
    // get image
    start = clock();
    start_all = start;
    Mat img(read_from_camera(pCapture), true); // copy data
    resize(img, img, input_size);
    CHECK_EQ(img.cols, image_w) << "image size mismatch";
    CHECK_EQ(img.rows, image_h) << "image size mismatch";
    finish = clock();
    LOG(INFO) << "Load image from camera: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    int proposal_num = bing_boxes(img, boxes, max_proposal_num);
    boxes2conv5(boxes, max_proposal_num, proposal_num, conv5_windows,
        conv5_scales);
    finish = clock();
    LOG(INFO) << "Run BING: " << (1000 * (finish - start) / CLOCKS_PER_SEC)
        << " ms, got " << proposal_num << " boxes";
    
    start = clock();
    Mat2float(image_data, img, channel_mean);
    finish = clock();
    LOG(INFO) << "Preprocess image: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    const float* result_vecs = forward_network(caffe_test_net, image_data,
        conv5_windows, conv5_scales, boxes, class_mask, class_num,
        max_proposal_num, img);
    finish = clock();
    LOG(INFO) << "Forward image: " << 1000 * (finish - start) / CLOCKS_PER_SEC
        << " ms";

    start = clock();
    draw_results(img, result_vecs, boxes, max_proposal_num, class_name_vec);
    imshow("detection results", img);
    finish = clock();
    finish_all = finish;
    LOG(INFO) << "Show result: " << 1000 * (finish - start) / CLOCKS_PER_SEC
        << " ms";
    LOG(INFO) << "Total Time: "
        << 1000 * (finish_all - start_all) / CLOCKS_PER_SEC << " ms";
    LOG(INFO) << "Frame Rate: "
        << CLOCKS_PER_SEC / float(finish_all - start_all) << " fps";
  }
  return 0;
}

// get a 640 by 480 demo
const IplImage* read_from_camera(CvCapture* pCapture) {
  IplImage* pFrame = cvQueryFrame(pCapture);
  waitKey(20);
  return pFrame;
}

// do mean subtraction and convert into float type
// Mat is already in BGR channel
void Mat2float(float image_data[], const Mat& img, const float channel_mean[]) {
  const int channels = 3;
  const int width = img.cols;
  const int height = img.rows;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        float pixel = static_cast<float>(img.at<cv::Vec3b>(h, w)[c]);
        image_data[w + width * (h + height * c)] = pixel - channel_mean[c];
      }
    }
  }
}

void load_channel_mean(float channel_mean[], const char* filename) {
  ifstream fin(filename, ios::binary);
  CHECK(fin) << "cannot open channel mean file";
  fin.read(reinterpret_cast<char*>(channel_mean), 3*sizeof(float));
  CHECK(fin) << "error reading channel mean file";
  fin.close();
  LOG(INFO) << "Channel mean: B = " << channel_mean[0]
      << ", G = " << channel_mean[1] << ", R = " << channel_mean[2];
}

void load_class_name(vector<string>& class_name_vec, const char* filename) {
  ifstream fin(filename);
  CHECK(fin) << "cannot open class name file";
  for (int i = 0; i < class_name_vec.size(); i++) {
    getline(fin, class_name_vec[i]);
    LOG(INFO) << "loading class: " << class_name_vec[i];
  }
  CHECK(fin) << "error loading class name file";
  fin.close();
}

void load_class_mask(float class_mask[]) {
  for (int i = 0; i < 7405; i++)
    class_mask[i] = 1;
}

const float* forward_network(Net<float>& net, float image_data[],
    float conv5_windows[], float conv5_scales[], float boxes[],
    float class_mask[], const int class_num, const int max_proposal_num,
    const Mat& img) {
  vector<Blob<float>*>& input_blobs = net.input_blobs();
  CHECK_EQ(input_blobs[0]->count(), img.rows*img.cols*3)
      << "input image_data mismatch";
  CHECK_EQ(input_blobs[1]->count(), max_proposal_num*4)
      << "input conv5_windows mismatch";
  CHECK_EQ(input_blobs[2]->count(), max_proposal_num)
      << "input conv5_scales mismatch";
  CHECK_EQ(input_blobs[3]->count(), max_proposal_num*4)
      << "input boxes mismatch";
  CHECK_EQ(input_blobs[4]->count(), class_num)
      << "input class_mask mismatch";
  memcpy(input_blobs[0]->mutable_cpu_data(), image_data,
      sizeof(float) * input_blobs[0]->count());
  memcpy(input_blobs[1]->mutable_cpu_data(), conv5_windows,
      sizeof(float) * input_blobs[1]->count());
  memcpy(input_blobs[2]->mutable_cpu_data(), conv5_scales,
      sizeof(float) * input_blobs[2]->count());
  memcpy(input_blobs[3]->mutable_cpu_data(), boxes,
      sizeof(float) * input_blobs[3]->count());
  memcpy(input_blobs[4]->mutable_cpu_data(), class_mask,
      sizeof(float) * input_blobs[4]->count());
  
  const vector<Blob<float>*>& result = net.ForwardPrefilled();
  CHECK_EQ(result[0]->count(), 3 * max_proposal_num)
      << "input class_mask mismatch";
  return result[0]->cpu_data();
}

void draw_results(Mat& img, const float result_vecs[], float boxes[],
    int max_proposal_num, vector<string>& class_name_vec) {
  const static CvScalar color = cvScalar(0, 0, 255);
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, CV_AA);
  const float* keep_vec = result_vecs;
  const float* class_id_vec = result_vecs + max_proposal_num;
  const float* score_vec = result_vecs + max_proposal_num*2;
  char label[200];
  
  int obj_num = 0;
  for (int box_id = 0; box_id < max_proposal_num; box_id++) {
    if (keep_vec[box_id]) {
      int y1 = boxes[box_id*4  ];
      int x1 = boxes[box_id*4+1];
      int y2 = boxes[box_id*4+2];
      int x2 = boxes[box_id*4+3];
      int class_id = class_id_vec[box_id];
      float score = score_vec[box_id];
      sprintf(label, "%s: %.3f", class_name_vec[class_id].c_str(), score);
      Point ul(x1, y1), ur(x2, y1), ll(x1, y2), lr(x2, y2);
      line(img, ul, ur, color, 3);
      line(img, ur, lr, color, 3);
      line(img, lr, ll, color, 3);
      line(img, ll, ul, color, 3);
      IplImage iplimage = img;
      cvPutText(&iplimage, label, cvPoint(x1, y1 - 3), &font, CV_RGB(255,0,0));
      LOG(INFO) << "(x1,y1,x2,y2) = (" << x1 << "," << y1 << "," << x2 << ","
          << y2 << "): " << label;
      obj_num++;
    }
  }
  LOG(INFO) << "Found " << obj_num << " objects";
}
