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
#include <cfloat>
#include <string>

using namespace caffe;
using namespace cv;
using namespace std;

// function declarations
const IplImage* read_from_camera(CvCapture* pCapture);
void get_multiscale_image_data(float image_data[], const Mat& img,
    const float channel_mean[], int input_h, int input_w, int scale_num,
    const int* resized_h_arr, const int* resized_w_arr);
void get_multiscale_conv5(float conv5_windows[], float conv5_scales[],
    float valid_vec[], const float boxes[], const int max_proposal_num,
    const int proposal_num, int original_h, int original_w, int scale_num,
    const int* resized_h_arr, const int* resized_w_arr);
void load_channel_mean(float channel_mean[], const char* filename);
void load_class_name(vector<string>& class_name_vec, const char* filename);
void draw_results(Mat& img, const float keep_vec[], const float class_id_vec[], 
    const float score_vec[], float boxes[], int max_proposal_num,
    vector<string>& class_name_vec);
    
struct PrefetchParameterSet {
  CvCapture* pCapture;
  int max_proposal_num;
  int class_num;
  int input_h;
  int input_w;
  int scale_num;
  const int* resized_h_arr;
  const int* resized_w_arr;
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
  int input_h = prefetch_param_ptr->input_h;
  int input_w = prefetch_param_ptr->input_w;
  int scale_num = prefetch_param_ptr->scale_num;
  const int* resized_h_arr = prefetch_param_ptr->resized_h_arr;
  const int* resized_w_arr = prefetch_param_ptr->resized_w_arr;
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
  Mat img(read_from_camera(pCapture), false); // do not copy data
  img.copyTo(*img_ptr); // copy data here
  const int original_h = img.rows; // should be 480
  const int original_w = img.cols; // should be 640
  
  // run objectness on the 640x480 original image (not rescaled images)
  int proposal_num = window_proposal(img, boxes_fetch, max_proposal_num);
  
  // resize image, and convert resized images to float-point data
  get_multiscale_image_data(image_data, img, channel_mean, input_h, input_w,
      scale_num, resized_h_arr, resized_w_arr);
  
  // match the boxes to conv5_windows and conv5_scales
  get_multiscale_conv5(conv5_windows, conv5_scales, valid_vec, boxes_fetch,
      max_proposal_num, proposal_num, original_h, original_w, scale_num,
      resized_h_arr, resized_w_arr);
  return (void*)0;
}

int main(int argc, char** argv) {
  CHECK_EQ(argc, 7) << "usage: lsda_demo.bin [net prototext] [net protobinary]"
      << " [channel mean file] [class name file] [scale num] [gpu id]"
  
  // general parameters
  CvCapture* pCapture = cvCreateCameraCapture(0);
  const int max_proposal_num = 1000;
  const int class_num = 7604;
  const int scale_num = atoi(argv[5]);
  const int device_id = atoi(argv[6]);
  const int channels = 3;
  int input_h;
  int input_w;
  const int* resized_h_arr;
  const int* resized_w_arr;
  
  // 5-scale parameters
  const int input_h_5_scale = 1200;
  const int input_w_5_scale = 1600;
  const int resized_h_arr_5_scale[] = { 480,  576,  688,  864,  1200};
  const int resized_w_arr_5_scale[] = { 640,  768,  917,  1152, 1600};
  // 1-scale parameters
  const int input_h_1_scale = 688;
  const int input_w_1_scale = 917;
  const int resized_h_arr_1_scale[] = { 688};
  const int resized_w_arr_1_scale[] = { 917};
  
  // match_scale
  switch (scale_num) {
    case 5:
      input_h = input_h_5_scale;
      input_w = input_w_5_scale;
      resized_h_arr = resized_h_arr_5_scale;
      resized_w_arr = resized_w_arr_5_scale;
      break;
    case 1:
      input_h = input_h_1_scale;
      input_w = input_w_1_scale;
      resized_h_arr = resized_h_arr_1_scale;
      resized_w_arr = resized_w_arr_1_scale;
      break;
    default:
      LOG(FATAL) << "Invalid scale_num: " << scale_num
          << ". scale_num must be either 1 or 5";
      break;
  }
  
  // Storage
  float* image_data = new float[input_h*input_w*channels*scale_num];
  float boxes_fetch[max_proposal_num*4];
  float boxes_show[max_proposal_num*4];
  float conv5_windows[max_proposal_num*4];
  float conv5_scales[max_proposal_num];
  float valid_vec[max_proposal_num];
  float channel_mean[channels];
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
  
  prefetch_param.input_h = input_h;
  prefetch_param.input_w = input_w;
  prefetch_param.scale_num = scale_num;
  prefetch_param.resized_h_arr = resized_h_arr;
  prefetch_param.resized_w_arr = resized_w_arr;
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
  CHECK_EQ(input_blobs[0]->count(), input_h * input_w * channels * scale_num)
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
        input_blobs[3]->mutable_cpu_data()); // To CPU
    caffe_copy(input_blobs[4]->count(), valid_vec,
        input_blobs[4]->mutable_cpu_data()); // To CPU
    // create prefetch thread
    CHECK(!pthread_create(&fetch_thread, NULL, prefetchThread, &prefetch_param))
        << "Failed to create prefetch thread";
    finish = clock();
    LOG(INFO) << "LSDA: fetch data and load data to gpu: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";

    start = clock();
    // forward network
    caffe_test_net.ForwardPrefilled();
    finish = clock();
    LOG(INFO) << "LSDA: forward image: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    // retrieve data from gpu
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->cpu_data(),
        result_vecs);
    finish = clock();
    LOG(INFO) << "LSDA: retrieve data from gpu: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    start = clock();
    // draw results
    draw_results(img_show, keep_vec, class_id_vec, score_vec, boxes_show,
        max_proposal_num, class_name_vec);
    imshow("LSDA Detection Results", img_show);
    waitKey(40);
    finish = clock();
    LOG(INFO) << "LSDA: show result: "
        << 1000 * (finish - start) / CLOCKS_PER_SEC << " ms";
    
    // Estimate FPS
    finish_all = clock();
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

void get_multiscale_conv5(float conv5_windows[], float conv5_scales[],
    float valid_vec[], const float boxes[], const int max_proposal_num,
    const int proposal_num, int original_h, int original_w, int scale_num,
    const int* resized_h_arr, const int* resized_w_arr) {
  const int conv5_stride = 16;
  // zero out all data
  memset(conv5_windows, 0, max_proposal_num * 4 * sizeof(float));
  memset(conv5_scales, 0, max_proposal_num * sizeof(float));
  memset(valid_vec, 0, max_proposal_num * sizeof(float));
  // calculate the corresponing windows on conv5 feature map
  const float desired_area = 50176; // 224 * 224;
  for (int i = 0; i < proposal_num; i++) {
    float y1 = boxes[4*i];
    float x1 = boxes[4*i+1];
    float y2 = boxes[4*i+2];
    float x2 = boxes[4*i+3];
    // find the best matching scale
    float area = (y2 - y1 + 1) * (x2 - x1 + 1);
    float min_area_diff = FLT_MAX;
    int matching_scale = -1;
    float matching_zoom_h = 0, matching_zoom_w = 0;
    for (int scale = 0; scale < scale_num; scale++) {
      const int resized_h = resized_h_arr[scale];
      const int resized_w = resized_w_arr[scale];
      const float zoom_h =
        static_cast<float>(resized_h) / static_cast<float>(original_h);
      const float zoom_w =
        static_cast<float>(resized_w) / static_cast<float>(original_w);
      float zoomed_area = area * zoom_h * zoom_w;
      float area_diff = abs(zoomed_area - desired_area);
      if (area_diff < min_area_diff) {
        min_area_diff = area_diff;
        matching_scale = scale;
        matching_zoom_h = zoom_h;
        matching_zoom_w = zoom_w;
      }
    }
    conv5_scales[i] = matching_scale;
    // LOG(INFO) << "scale: " << matching_scale << " zoom_h: "
    //     << matching_zoom_h << " zoom_w: " << matching_zoom_w;
    // round and add 1 to ends
    conv5_windows[4*i  ] =
        static_cast<int>(0.5f + y1 * matching_zoom_h / conv5_stride);
    conv5_windows[4*i+1] =
        static_cast<int>(0.5f + x1 * matching_zoom_w / conv5_stride);
    conv5_windows[4*i+2] =
        static_cast<int>(0.5f + y2 * matching_zoom_h / conv5_stride) + 1;
    conv5_windows[4*i+3] =
        static_cast<int>(0.5f + x2 * matching_zoom_w / conv5_stride) + 1;
    // set valid to be true
    valid_vec[i] = 1;
  }
}

// do mean subtraction and convert into float type
// Mat is already in BGR channel
void get_multiscale_image_data(float image_data[], const Mat& img,
    const float channel_mean[], int input_h, int input_w, int scale_num,
    const int* resized_h_arr, const int* resized_w_arr) {
  const int channels = 3;
  // image offset for the expanded image
  const int image_offset = input_h * input_w * channels;
  // zero out image data
  memset(image_data, 0, image_offset * scale_num * sizeof(float));
  // resize image and perform mean subtraction
  Mat resized_img;
  for (int scale = 0; scale < scale_num; scale++) {
    const int resized_h = resized_h_arr[scale];
    const int resized_w = resized_w_arr[scale];
    resize(img, resized_img, Size(resized_w, resized_h));
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < resized_h; ++h) {
        for (int w = 0; w < resized_w; ++w) {
          float pixel = static_cast<float>(resized_img.at<cv::Vec3b>(h, w)[c]);
          int index = (image_offset * scale)
              + (w + input_w * (h + input_h * c));
          image_data[index] = pixel - channel_mean[c];
        }
      }
    }
  }
}

void load_channel_mean(float channel_mean[], const char* filename) {
  const int channels = 3;
  ifstream fin(filename, ios::binary);
  CHECK(fin) << "cannot open channel mean file";
  fin.read(reinterpret_cast<char*>(channel_mean), channels * sizeof(float));
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
  const int line_width = 3;
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
      line(img, ul, ur, (class_id < strong_cls_num ? blu : red), line_width);
      line(img, ur, lr, (class_id < strong_cls_num ? blu : red), line_width);
      line(img, lr, ll, (class_id < strong_cls_num ? blu : red), line_width);
      line(img, ll, ul, (class_id < strong_cls_num ? blu : red), line_width);
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
