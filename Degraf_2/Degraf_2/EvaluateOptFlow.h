// Header file for the heavily modified version of the openCV optical_flow_evaluation.cpp file:
// https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_evaluation.cpp 
// Adapted to be able to evaluate KITTI and Middlebury datasets.

#pragma once

#include "SaliencyDetector.h"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>

#include <iostream>		// standard C++ I/O
#include <string>		// standard C++ I/O
#include <algorithm>    // includes max()
#include <vector>

using namespace cv;

class EvaluateOptFlow {

public:

	// Stats for a single flow computation 
	vector<float> stats_vector;

	// List of stats over all image pairs 
	vector<vector<float>> all_stats;

	EvaluateOptFlow();

	/*inline bool isFlowCorrect(const Point2f u);
	inline bool isFlowCorrect(const Point3f u);
	static Mat endpointError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2);
	static Mat angularError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2);
	static float stat_RX(Mat errors, float threshold, Mat mask);
	static float stat_AX(Mat hist, int cutoff_count, float max_value);
	static void calculateStats(Mat errors, Mat mask = Mat(), bool display_images = false);
	static Mat flowToDisplay(const Mat flow);*/

	void calculateStats(Mat errors, Mat mask, bool display_images); // adding this as public so it can update the stats_vector variable 

	int EvaluateOptFlow::runEvaluation(String method, bool display, int image_no);
};
