wget http://googledrive.com/host/0B6qziMs8hVGieFg0UzE0WmZaOW8/code/gop_1.1.zip
unzip gop_1.1.zip

cp -r gop_wrapper/* gop_1.1/
cp ../include/caffe/util/window_proposer.hpp gop_1.1/examples/
