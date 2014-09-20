include_directories( ../lib )

add_library( gop gop_window_proposer.cpp)
target_link_libraries( gop imgproc proposals contour opencv_core opencv_highgui opencv_imgproc)
