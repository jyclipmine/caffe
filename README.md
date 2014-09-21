# Caffe LSDA Release Installation Guide

Let's call the place where you installed caffe-lsda-nips `$CAFFE_ROOT` (you can run `export CAFFE_ROOT=$(pwd)`)

0. Download and build Geodesic Object Proposal (GOP) libraries
  0. Make sure you have installed all GOP Prerequisites:
    * cmake
    * g++-4.8 (or other compliers supporting C++11)
    * Eigen3
    * libpng
    * libjpeg   
    on Ubuntu, you can install them using apt-get:    
          `sudo apt-get install cmake g++-4.8 libeigen3-dev libpng-dev libjpeg-dev`
  0. Run the following commands to download and build Geodesic Object Proposal (GOP) libraries 
    `cd $CAFFE_ROOT/lib`
    `./download_gop.sh`
    `./build_gop_libraries.sh`
0. Build Caffe LSDA Release
  Please Follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)
  Also, make sure that your CUDA version supports g++-4.8 (or the compile you are using to compile gop). It is highly recommended to complie caffe with latest CUDA 6.5.

0. Try out LSDA example
  0. Download the lsda caffe model    
    `cd $CAFFE_ROOT/examples/lsda-nips`   
    `./download_model.sh`
  1. Try out the detection example  
    1-scale Spatial Pyramid Pooling (fast detection):   
    `./lsda_1_scale.sh`   
    5-scale Spatial Pyramid Pooling (slower than 1-scale but more accurate):    
    `./lsda_5_scale.sh`   
    If you want to run on your own image, you can simply modify `lsda_1_scale.sh` or `lsda_5_scale.sh` and replace `-impath input.jpg` with path to your own image.
