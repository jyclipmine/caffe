#include "kyheader.h"
#include "DataSetVOC.h"

DataSetVOC::DataSetVOC(CStr &_wkDir)
{
	wkDir = _wkDir;
	resDir = wkDir + "Results/";
	localDir = wkDir + "Local/";
	imgPathW = wkDir + "JPEGImages/%s.jpg";
	annoPathW = wkDir + "Annotations/%s.yml";
//	CmFile::MkDir(resDir);
//	CmFile::MkDir(localDir);

    trainSet = CmFile::loadStrList(wkDir + "ImageSets/Main/train.txt");
    testSet = CmFile::loadStrList(wkDir + "ImageSets/Main/test.txt");
	classNames = CmFile::loadStrList(wkDir + "ImageSets/Main/class.txt");

	trainNum = trainSet.size();
	testNum = testSet.size();
}


Vec4i getMaskRange(CMat &mask1u, int ext = 0)
{
	int maxX = INT_MIN, maxY = INT_MIN, minX = INT_MAX, minY = INT_MAX, rows = mask1u.rows, cols = mask1u.cols;
	for (int r = 0; r < rows; r++)	{
		const byte* data = mask1u.ptr<byte>(r);
		for (int c = 0; c < cols; c++)
			if (data[c] > 10) {
				maxX = max(maxX, c);
				minX = min(minX, c);
				maxY = max(maxY, r);
				minY = min(minY, r);
			}
	}

	maxX = maxX + ext + 1 < cols ? maxX + ext + 1 : cols;
	maxY = maxY + ext + 1 < rows ? maxY + ext + 1 : rows;
	minX = minX - ext > 0 ? minX - ext : 0;
	minY = minY - ext > 0 ? minY - ext : 0;

	return Vec4i(minX + 1, minY + 1, maxX, maxY); // Rect(minX, minY, maxX - minX, maxY - minY);
}

void DataSetVOC::loadAnnotations()
{
	gtTrainBoxes.resize(trainNum);
	gtTrainClsIdx.resize(trainNum);
	for (int i = 0; i < trainNum; i++)
		if (!loadBBoxes(trainSet[i], gtTrainBoxes[i], gtTrainClsIdx[i]))
			return;

	gtTestBoxes.resize(testNum);
	gtTestClsIdx.resize(testNum);
	for (int i = 0; i < testNum; i++)
		if(!loadBBoxes(testSet[i], gtTestBoxes[i], gtTestClsIdx[i]))
			return;
	printf("Load annotations finished\n");
}

void DataSetVOC::loadBox(const FileNode &fn, vector<Vec4i> &boxes, vecI &clsIdx){
	string isDifficult;
	fn["difficult"]>>isDifficult;
	if (isDifficult == "1")
		return; 

	string strXmin, strYmin, strXmax, strYmax;
	fn["bndbox"]["xmin"] >> strXmin;
	fn["bndbox"]["ymin"] >> strYmin;
	fn["bndbox"]["xmax"] >> strXmax;
	fn["bndbox"]["ymax"] >> strYmax;
	boxes.push_back(Vec4i(atoi(_S(strXmin)), atoi(_S(strYmin)), atoi(_S(strXmax)), atoi(_S(strYmax))));

	string clsName;
	fn["name"]>>clsName;
	clsIdx.push_back(findFromList(clsName, classNames));
	CV_Assert_(clsIdx[clsIdx.size() - 1] >= 0, ("Invalidate class name\n"));
}

bool DataSetVOC::loadBBoxes(CStr &nameNE, vector<Vec4i> &boxes, vecI &clsIdx)
{
	string fName = format(_S(annoPathW), _S(nameNE));
	FileStorage fs(fName, FileStorage::READ);
	FileNode fn = fs["annotation"]["object"];
	boxes.clear();
	clsIdx.clear();
	if (fn.isSeq()){
        for (FileNodeIterator it = fn.begin(), it_end = fn.end(); it != it_end; it++){
            loadBox(*it, boxes, clsIdx);
        }
	}
	else
		loadBox(fn, boxes, clsIdx);
	return true;
}
