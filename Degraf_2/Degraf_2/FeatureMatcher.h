/*!
\file FeatureMatcher.h
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow
\author Felix Stephenson
*/

#pragma once

#include "GradientDetector.h"
#include "SaliencyDetector.h"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"

#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"

#include <iostream>		// standard C++ I/O
#include <string>		// standard C++ I/O
#include <algorithm>    // includes max()
#include <vector>

// N.B need RLOF code from https://github.com/tsenst/RLOFLib
#include <RLOF_Flow.h>

using namespace cv;

class FeatureMatcher {

	public:
		// Public variables
		vector<Point2f> points_filtered, dst_points_filtered; // corresponding points in each image

		// Public functions
		FeatureMatcher();
		void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma);
		void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma);
};
