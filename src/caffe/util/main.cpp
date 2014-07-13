#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"

void RunBING(Mat& img)
{
	int base = 2, W = 8, NSS = 2, numPerSz = 250;
	DataSetVOC dataset;
	Objectness objNess(dataset, base, W, NSS);
	ValStructVec<float, Vec4i> boxesTests;
	objNess.getObjBndBoxesSingleImg(img, boxesTests, numPerSz);
	for (int i = 0; i < boxesTests.size(); i++) {
		// output order: y1 x1 y2 x2
		cout << boxesTests[i][1] << " " << boxesTests[i][0]
		  << " " << boxesTests[i][3] << " " << boxesTests[i][2] << endl;
	}
}

void RunVOC(double base, int W, int NSS, int numPerSz)
{
	srand((unsigned int)time(NULL));
	DataSetVOC voc2007("/home/ronghang/workspace/VOC2007/");
	voc2007.loadAnnotations();

	printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
	printf("Base = %g, W = %d, NSS = %d, perSz = %d\n", base, W, NSS, numPerSz);

	Objectness objNess(voc2007, base, W, NSS);
	vector<vector<Vec4i> > boxesTests;
	objNess.getObjBndBoxesForTests(boxesTests, numPerSz);
}
