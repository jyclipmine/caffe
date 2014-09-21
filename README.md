# LSDA

[LSDA](http://lsda.berkeleyvision.org/) is a framework for large scale detection through adaptation. We combine adaptation techniques with deep convolutional models to create a fast and effective large scale detection network.

This git repository contains the fast LSDA implemenation using [Geodesic Object Proposal (GOP)](http://www.philkr.net/home/gop) and [Spatial Pyramid Pooling (SPP)](http://research.microsoft.com/en-us/um/people/kahe/eccv14sppnet/index.html).

## Caffe LSDA Release Installation Guide

0. Download Caffe LSDA Release (this repo):    
  `wget --no-check-certificate https://github.com/ronghanghu/caffe/archive/lsda-nips.zip`    
  `unzip lsda-nips.zip`   
  Let's call the path to caffe-lsda-nips `$CAFFE_ROOT` (you can run `export CAFFE_ROOT=$(pwd)/caffe-lsda-nips`)
0. Download and build Geodesic Object Proposal (GOP) libraries
  0. Make sure you have installed all GOP prerequisites:
    * cmake
    * g++-4.8 (or other compliers supporting C++11, see Note 1)
      * do NOT use g++ 4.8.1 due to its [fatal bug](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57532), use g++ 4.8.2 or higher.
    * [Eigen3](http://eigen.tuxfamily.org/)
    * libpng
    * libjpeg   
    on Ubuntu (tested on 14.04 LTS), you can install them using apt-get:    
          `sudo apt-get install cmake g++-4.8 libeigen3-dev libpng-dev libjpeg-dev`   
    If you are using the a Ubuntu version lower than 14.04 and installation script does not work, see Note 2.
  0. Run the following commands to download and build Geodesic Object Proposal (GOP) libraries:    
    `cd $CAFFE_ROOT/lib`    
    `./download_gop.sh`   
    `./build_gop_libraries.sh`    
0. Build Caffe LSDA Release     
  Please follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html).    
  Also, make sure that your CUDA version supports g++-4.8 (or the compiler you are using to compile GOP, see Note 1). It is highly recommended to complie caffe with latest CUDA 6.5.

0. Try out LSDA example
  0. Download the lsda caffe model:    
    `cd $CAFFE_ROOT/examples/lsda-nips`   
    `./download_model.sh`
  0. Enjoy the car detection example
    * 5-scale Spatial Pyramid Pooling:    
    `./lsda_5_scale.sh`   
    * 1-scale Spatial Pyramid Pooling (faster than 5-scale but less accurate):   
    `./lsda_1_scale.sh`   
    If you want to run on your own image, you can simply modify `lsda_1_scale.sh` or `lsda_5_scale.sh` and replace `-impath input.jpg` with path to your own image.

Note 1: To link the GOP libraries and caffe libraries together, they must be compiled using the same C++ compiler with C++11 support. If you want to use a compiler other than g++-4.8, you must modify the first line in `$CAFFE_ROOT/Makefile` and the first line in `$CAFFE_ROOT/lib/gop_wrapper/CMakeLists.txt`. Also make sure that CUDA nvcc supports your compiler.   
Note 2: If the script to install GOP dependencies does not work on early Ubuntu version (e.g. 12.04 LTS), you may add the Ubuntu 14.04 package source to your `/etc/apt/sources.list`. For example, if you want to use the UC Davis source, you can run:   
`sudo echo "deb http://mirror.math.ucdavis.edu/ubuntu/ trusty main restricted" >> /etc/apt/sources.list`    
`sudo apt-get update`    
Or, you can install the GOP dependencies manually.
