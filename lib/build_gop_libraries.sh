mkdir gop_1.1/build
cd gop_1.1/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

mkdir ../../gop_lib/
find . -name "*.a" -exec cp {} ../../gop_lib/ \;
cd ../..
