#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include <algorithm>
#include "caffe/common.hpp"

namespace caffe {

// the returned boxes are 0-indexed, [y1 x1 y2 x2]
int window_proposal_bing(const Mat& img, float boxes[],
    const int max_proposal_num) {
	// Parameters of BING
	const int max_size = 600, min_size = 80, im_width = 500;
	static DataSetVOC dataset;
	int base = 2, W = 8, NSS = 2, numPerSz = 250;
	static Objectness objNess(dataset, base, W, NSS);
	static ValStructVec<float, Vec4i> boxesTests;
  // resize the image to have a fixed width
  const int width = img.cols;
  const int height = img.rows;
	const float scale = static_cast<float>(width) / im_width;
	const Size bing_size(width / scale, height / scale);
	Mat resized_img;
	resize(img, resized_img, bing_size);
	// run BING to get boxes
	boxesTests.clear();
	objNess.getObjBndBoxesSingleImg(resized_img, boxesTests, numPerSz);
	// put the boxes into return array
	memset(boxes, 0, max_proposal_num * 4 * sizeof(float));
	int proposal_num = 0;
	for (int i = 0; i < boxesTests.size(); i++) {
	  // convert the output to 0-indexed
	  float x1 = (boxesTests[i][0] - 1) * scale;
	  float y1 = (boxesTests[i][1] - 1) * scale;
	  float x2 = (boxesTests[i][2] - 1) * scale;
	  float y2 = (boxesTests[i][3] - 1) * scale;
	  bool valid = (x2 - x1 >= min_size) && (x2 - x1 <= max_size)
	      && (y2 - y1 >= min_size) && (y2 - y1 <= max_size);
	  if (valid) {
	    boxes[4*proposal_num  ] = y1;
	    boxes[4*proposal_num+1] = x1;
	    boxes[4*proposal_num+2] = y2;
	    boxes[4*proposal_num+3] = x2;
	    proposal_num++;
	    if (proposal_num >= max_proposal_num)
	      break;
	  }
	}
	return proposal_num;
}

}  // namespace caffe 
