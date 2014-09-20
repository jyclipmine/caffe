mkdir gop_1.1/build
cd gop_1.1/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

mkdir ../../gop_lib/
cp examples/*.a ../../gop_lib/
cp lib/*/*.a ../../gop_lib/
