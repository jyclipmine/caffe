cp -r gop_wrapper/* gop_1.1/
cp ../include/caffe/util/window_proposer.hpp gop_1.1/examples/
cp data/sf.dat ../examples/lsda-nips/data/

mkdir gop_1.1/build
cd gop_1.1/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

mkdir ../../gop_lib/
find . -name "*.a" -exec cp {} ../../gop_lib/ \;
cd ../..
