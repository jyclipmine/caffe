/*
 * matlab.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: ronghang
 */

#include "kyheader.h"
#include "Objectness.h"
#include "CmShow.h"
#include "DataSetVOC.h"
#include <iostream>

DataSetVOC::DataSetVOC()
{
	resDir = "svms/";
}

void Objectness::getObjBndBoxesSingleImg(CMat& img, ValStructVec<float, Vec4i> &boxesTests, int numDetPerSize)
{
  cout << "getObjBndBoxesSingleImg start" << endl;
	//ValStructVec<float, Vec4i> boxesTests;
	boxesTests.clear();
	int scales[3] = {1, 3, 5};
	for (int clr = MAXBGR; clr <= G; clr++){
		setColorSpace(clr);
		loadTrainedModel();
		ValStructVec<float, Vec4i> boxes;
		getObjBndBoxes(img, boxes, numDetPerSize);
		boxesTests.append(boxes, scales[clr]);
	}
	boxesTests.sort(false);
	cout << "getObjBndBoxesSingleImg over" << endl;
}


