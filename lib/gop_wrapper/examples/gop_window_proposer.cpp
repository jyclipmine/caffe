#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"
#include "proposals/proposal.h"
#include "segmentation/segmentation.h"
#include "imgproc/image.h"
#include "window_proposer.hpp"

namespace caffe {

struct GOPParameters {
  ProposalSettings* prop_settings_ptr;
  Proposal* prop_ptr;
  StructuredForest* detector_ptr;
};

GOPWindowProposer::GOPWindowProposer() {
  GOPParameters* gop_parameters = new GOPParameters();
  parameters_ = gop_parameters;
  
	/* Setup the proposals settings */
	// Play with those numbers to increase the number of proposals
	// Number of seeds N_S and number of segmentations per seed N_T
	const int N_S = 130, N_T = 5;
	// Maximal overlap between any two proposals (intersection / union)
	const float max_iou = 0.8;
	
  gop_parameters->prop_settings_ptr = new ProposalSettings();
	gop_parameters->prop_settings_ptr->max_iou = max_iou;
	// Foreground/background proposals
	std::vector<int> vbg = {0,15};
	gop_parameters->prop_settings_ptr->unaries.push_back( ProposalSettings::UnarySettings( N_S, N_T, seedUnary(), backgroundUnary(vbg) ) );
	// Pure background proposals
	std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	gop_parameters->prop_settings_ptr->unaries.push_back( ProposalSettings::UnarySettings( 0, N_T, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );
	
	/* Create the proposlas */
	gop_parameters->prop_ptr = new Proposal( *(gop_parameters->prop_settings_ptr) );
	
	// Muilti Scale Structured Forests generally perform best
	gop_parameters->detector_ptr = new StructuredForest();
	gop_parameters->detector_ptr->load( "data/sf.dat" );
}

GOPWindowProposer::~GOPWindowProposer() {
  // Do nothing
  /*
  GOPParameters* gop_parameters = reinterpret_cast<GOPParameters*>(parameters_);
  delete gop_parameters->prop_settings_ptr;
  delete gop_parameters->prop_ptr;
  delete gop_parameters->detector_ptr;
  delete gop_parameters;
  */
}

int GOPWindowProposer::propose(const cv::Mat& img, float boxes[],
    const int max_proposal_num) {
  GOPParameters* gop_parameters = reinterpret_cast<GOPParameters*>(parameters_);
	
	/* Create the proposlas */
	Proposal& prop = *(gop_parameters->prop_ptr);
	StructuredForest& detector = *(gop_parameters->detector_ptr);

	// Load an image and create an over-segmentation
	Image8u im = fromMat(img);
	ImageOverSegmentation s = geodesicKMeans( im, detector, 1000 );
	RMatrixXb p = prop.propose( s );
	int raw_proposal_num = p.rows();
	
	// boxes are [x1 y1 x2 y2], 0-indexed
	RMatrixXi gop_boxes = s.maskToBox( p );
  
	memset(boxes, 0, max_proposal_num * 4 * sizeof(float));
	int proposal_num = 0;
  for (int i = 0; i < raw_proposal_num; i++) {
	  float x1 = gop_boxes(i, 0);
	  float y1 = gop_boxes(i, 1);
	  float x2 = gop_boxes(i, 2);
	  float y2 = gop_boxes(i, 3);
    boxes[4*proposal_num  ] = y1;
    boxes[4*proposal_num+1] = x1;
    boxes[4*proposal_num+2] = y2;
    boxes[4*proposal_num+3] = x2;
    proposal_num++;
    if (proposal_num >= max_proposal_num)
      break;
  }
	return proposal_num;
}

}
