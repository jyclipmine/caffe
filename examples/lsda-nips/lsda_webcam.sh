#!/usr/bin/env sh

../../build/tools/lsda_webcam.bin \
  -model data/lsda_1_scale.prototxt \
  -weights data/lsda_net.caffemodel \
  -channelmean data/channelmean.dat \
  -classname data/classname.txt \
  -scale 1 \
  -gpu 0 \
  -camera_id 0
