#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"
#include "caffe/util/window_proposal.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <string>

using namespace caffe; // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

// function declarations
void boxes2conv5(const float boxes[], const int max_proposal_num,
    const int proposal_num, const int original_h, const int original_w,
    const int resize_h, const int resize_w, float conv5_windows[],
    float conv5_scales[], float valid_vec[]);
const IplImage* read_from_camera(CvCapture* pCapture);
void Mat2float(float image_data[], const Mat& img, const float channel_mean[]);
void load_channel_mean(float channel_mean[], const char* filename);
void load_class_name(vector<string>& class_name_vec, const char* filename);
void draw_results(Mat& img, const float keep_vec[], const float class_id_vec[], 
    const float score_vec[], float boxes[], int max_proposal_num,
    vector<string>& class_name_vec);
    
struct PrefetchParameterSet {
  CvCapture* pCapture;
  int max_proposal_num;
  int class_num;
  Size input_size;
  float* image_data;
  float* conv5_windows;
  float* conv5_scales;
  float* boxes_fetch;
  float* valid_vec;
  float* channel_mean;
  Mat* img_ptr; // 640x480, copied from the original image from camra
  int (*window_proposal)(const Mat& img, float boxes[],
      const int max_proposal_num);
};

void* prefetchThread(void* ptr) {
  PrefetchParameterSet* prefetch_param_ptr =
      reinterpret_cast<PrefetchParameterSet*>(ptr);
  
  // Unpack prefetch_param
  CvCapture* pCapture = prefetch_param_ptr->pCapture;
  int max_proposal_num = prefetch_param_ptr->max_proposal_num;
  // int class_num = prefetch_param_ptr->class_num;
  Size input_size = prefetch_param_ptr->input_size;
  float* image_data = prefetch_param_ptr->image_data;
  float* conv5_windows = prefetch_param_ptr->conv5_windows;
  float* conv5_scales = prefetch_param_ptr->conv5_scales;
  float* boxes_fetch = prefetch_param_ptr->boxes_fetch;
  float* valid_vec = prefetch_param_ptr->valid_vec;
  float* channel_mean = prefetch_param_ptr->channel_mean;
  Mat* img_ptr = prefetch_param_ptr->img_ptr;
  int (*window_proposal)(const Mat& img, float boxes[],
      const int max_proposal_num) = prefetch_param_ptr->window_proposal;
  
  // load image from camera
  Mat temp_img(read_from_camera(pCapture), false); // do not copy data
  temp_img.copyTo(*img_ptr); // copy data
  const int original_h = img_ptr->rows;
  const int original_w = img_ptr->cols;
  
  // run objectness on the 640x480 original image (not rescaled images)
  int proposal_num = window_proposal(*img_ptr, boxes_fetch, max_proposal_num);
  
  // resize image, and convert resized images to float-point data
  Mat resized_img;
  resize(*img_ptr, resized_img, input_size);
  Mat2float(image_data, resized_img, channel_mean);
  
  // match the boxes to conv5_windows and conv5_scales
  int resize_h = resized_img.rows;
  int resize_w = resized_img.cols;
  boxes2conv5(boxes_fetch, max_proposal_num, proposal_num, original_h, original_w,
      resize_h, resize_w, conv5_windows, conv5_scales, valid_vec);
      
  return (void*)0;
}

int main(int argc, char** argv) {
  CHECK_EQ(argc, 6) << "Input argument number mismatch";
  
  // Parameters
  CvCapture* pCapture = cvCreateCameraCapture(0);
  const int max_proposal_num = 1000;
  const int class_num = 7604;
  const int device_id = atoi(argv[5]);
  const int image_w = 917;
  const int image_h = 688;
  const Size input_size(image_w, image_h);
  
  // Storage
  float image_data[image_h*image_w*3];
  float boxes_fetch[max_proposal_num*4];
  float boxes_show[max_proposal_num*4];
  float conv5_windows[max_proposal_num*4];
  float conv5_scales[max_proposal_num];
  float valid_vec[max_proposal_num];
  float channel_mean[3];
  float result_vecs[max_proposal_num*3];
  const float* keep_vec = result_vecs;
  const float* class_id_vec = result_vecs + max_proposal_num;
  const float* score_vec = result_vecs + max_proposal_num * 2;
  Mat img_fetch, img_show;
  
  // Set prefetch_param
  PrefetchParameterSet prefetch_param;
  prefetch_param.pCapture = pCapture;
  prefetch_param.max_proposal_num = max_proposal_num;
  prefetch_param.class_num = class_num;
  prefetch_param.input_size = input_size;
  prefetch_param.image_data = image_data;
  prefetch_param.conv5_windows = conv5_windows;
  prefetch_param.conv5_scales = conv5_scales;
  prefetch_param.boxes_fetch = boxes_fetch;
  prefetch_param.valid_vec = valid_vec;
  prefetch_param.channel_mean = channel_mean;
  prefetch_param.img_ptr = &img_fetch;
  prefetch_param.window_proposal = &window_proposal_bing;

  // thread for fetching data
  pthread_t fetch_thread;
  // create prefetch thread
  CHECK(!pthread_create(&fetch_thread, NULL, prefetchThread, &prefetch_param))
      << "Failed to create prefetch thread";
  
  // Initialize network
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(device_id);
  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  vector<string> class_name_vec(class_num);
  vector<Blob<float>*>& input_blobs = caffe_test_net.input_blobs();
  vector<Blob<float>*>& output_blobs = caffe_test_net.output_blobs();
  CHECK_EQ(input_blobs[0]->count(), image_h * image_w *3)
      << "input image_data mismatch";
  CHECK_EQ(input_blobs[1]->count(), max_proposal_num*4)
      << "input conv5_windows mismatch";
  CHECK_EQ(input_blobs[2]->count(), max_proposal_num)
      << "input conv5_scales mismatch";
  CHECK_EQ(input_blobs[3]->count(), max_proposal_num*4)
      << "input boxes mismatch";
  CHECK_EQ(input_blobs[4]->count(), max_proposal_num)
      << "input valid_vec mismatch";
  CHECK_EQ(output_blobs[0]->count(), max_proposal_num*3)
      << "output size mismatch";
  
  // load data from disk
  load_channel_mean(channel_mean, argv[3]);
  load_class_name(class_name_vec, argv[4]);
  
  // timing
  clock_t start, finish;
  clock_t start_all, finish_all;
  // run loop
  while (true) {
    LOG(INFO) << "-------------------------------------------";
    start_all = clock();
    
    start = clock();
    // join prefetch thread
    CHECK(!pthread_join(fetch_thread, NULL))
        << "Failed to join prefetch thread";
    img_fetch.copyTo(img_show);
    memcpy(boxes_show, boxes_fetch, 4 * max_proposal_num * sizeof(float));
    // load data to gpu
    caffe_copy(input_blobs[0]->count(), image_data,
        input_blobs[0]->mutable_gpu_data());
    caffe_copy(input_blobs[1]->count(), conv5_windows,
        input_blobs[1]->mutable_gpu_data());
    caffe_copy(input_blobs[2]->count(), conv5_scales,
        input_blobs[2]->mutable_gpu_data());
    caffe_copy(input_blobs[3]->count(), boxes_fetch,
        input_blobs[3]->mutable_cpu_data());
    caffe_copy(input_blobs[4]->count(), valid_vec,
        input_blobs[4]->mutable_cpu_data());
    // create prefetch thread
    CHECK(!pthread_create(&fetch_thread, NULL, prefetchThread, &prefetch_param))
        << "Failed to create prefetch thread";
    finish = clock();
    LOG(INFO) << "Fetch data and load data to gpu: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";

    start = clock();
    // forward network
    caffe_test_net.ForwardPrefilled();
    finish = clock();
    LOG(INFO) << "Caffe forward image: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    // retrieve data from gpu
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->cpu_data(),
        result_vecs);
    finish = clock();
    LOG(INFO) << "Caffe retrieve data from gpu: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    // draw results
    draw_results(img_show, keep_vec, class_id_vec, score_vec, boxes_show,
        max_proposal_num, class_name_vec);
    imshow("detection results", img_show);
    waitKey(40);
    finish = clock();
    LOG(INFO) << "Show result: " << 1000 * (finish - start) / CLOCKS_PER_SEC
        << " ms";
    
    // Estimate FPS
    finish_all = clock();
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
  CHECK(pFrame)
      << "Failed to read image from camera. Check your camera settings";
  // wait some time so that the image is fully loaded
  waitKey(20);
  return pFrame;
}

void boxes2conv5(const float boxes[], const int max_proposal_num,
    const int proposal_num, const int original_h, const int original_w,
    const int resize_h, const int resize_w, float conv5_windows[],
    float conv5_scales[], float valid_vec[]) {
  const int conv5_stride = 16;
  const float zoom_h =
      static_cast<float>(resize_h) / static_cast<float>(original_h);
  const float zoom_w =
      static_cast<float>(resize_w) / static_cast<float>(original_w);

  // zero out boxes
  memset(conv5_windows, 0, max_proposal_num * 4 * sizeof(float));
  // zero out valid_vec
  memset(valid_vec, 0, max_proposal_num * sizeof(float));
  // calculate the corresponing windows on conv5 feature map
  for (int i = 0; i < proposal_num; i++) {
    float y1 = zoom_h * boxes[4*i];
    float x1 = zoom_w * boxes[4*i+1];
    float y2 = zoom_h * boxes[4*i+2];
    float x2 = zoom_w * boxes[4*i+3];
    // round and add 1 to ends
    conv5_windows[4*i  ] = static_cast<int>(0.5f + y1 / conv5_stride);
    conv5_windows[4*i+1] = static_cast<int>(0.5f + x1 / conv5_stride);
    conv5_windows[4*i+2] = static_cast<int>(0.5f + y2 / conv5_stride) + 1;
    conv5_windows[4*i+3] = static_cast<int>(0.5f + x2 / conv5_stride) + 1;
    // set valid to be true
    valid_vec[i] = 1;
  }
  // for now, set all scales to be zero
  memset(conv5_scales, 0, max_proposal_num * sizeof(float));
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
    LOG(INFO) << "loading class No. " << (i + 1) << ": " << class_name_vec[i];
  }
  CHECK(fin) << "error loading class name file";
  fin.close();
}

void draw_results(Mat& img, const float keep_vec[], const float class_id_vec[], 
    const float score_vec[], float boxes[], int max_proposal_num,
    vector<string>& class_name_vec) {
  const static CvScalar blu = cvScalar(255, 0, 0);
  const static CvScalar red = cvScalar(0, 0, 255);
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, CV_AA);
  char label[200];
  int obj_num = 0;
  const int strong_cls_num = 200;
  for (int box_id = 0; box_id < max_proposal_num; box_id++) {
    if (keep_vec[box_id]) {
      int y1 = static_cast<int>(boxes[box_id*4  ]);
      int x1 = static_cast<int>(boxes[box_id*4+1]);
      int y2 = static_cast<int>(boxes[box_id*4+2]);
      int x2 = static_cast<int>(boxes[box_id*4+3]);
      int class_id = static_cast<int>(class_id_vec[box_id]);
      float score = score_vec[box_id];
      sprintf(label, "%s: %.3f", class_name_vec[class_id].c_str(), score);
      Point ul(x1, y1), ur(x2, y1), ll(x1, y2), lr(x2, y2);
      line(img, ul, ur, (class_id < strong_cls_num ? blu : red), 3);
      line(img, ur, lr, (class_id < strong_cls_num ? blu : red), 3);
      line(img, lr, ll, (class_id < strong_cls_num ? blu : red), 3);
      line(img, ll, ul, (class_id < strong_cls_num ? blu : red), 3);
      IplImage iplimage = img;
      cvPutText(&iplimage, label, cvPoint(x1, y1 - 3), &font,
          (class_id < strong_cls_num ? CV_RGB(0, 0, 255) : CV_RGB(255, 0, 0)));
      LOG(INFO) << "box id " << box_id << ", (x1,y1,x2,y2) = (" << x1 << ","
          << y1 << "," << x2 << "," << y2 << "), No. " << (class_id + 1) << ": "
          << label;
      obj_num++;
    }
  }
  LOG(INFO) << "Found " << obj_num << " objects";
}
