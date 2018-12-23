/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow 
\author Felix Stephenson
*/

#include "stdafx.h"
#include "FeatureMatcher.h"

// Constructor 
FeatureMatcher::FeatureMatcher() {
}

// DeGraF-Flow using lucas-kanade point tracking
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/

void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));

	int64 timeStart0 = getTickCount();
	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	vector<Point2f> dst_points;
	vector<unsigned char> status;
	vector<float> err;

	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0) {
		cv::Size s = from.size();

		IplImage *dog_1 = cvCreateImage(cvSize(s.width, s.height), IPL_DEPTH_8U, 3);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(&(IplImage(from.getMat())), dog_1, 5, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();

		int status_1 = gradient_detector_1->DetectGradients(dog_1, 3, 3, 7, 7);

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);
	}
	// Using other point detectors
	else if(point == 1){
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints, 2, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2) {
		vector<KeyPoint> keypoints;

		Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(0.1,4,2);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3) {
		vector<KeyPoint> keypoints;
		Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(7000,3,0.00,100.0);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4) {
		vector<KeyPoint> keypoints;
		Ptr<FeatureDetector> detector = ORB::create(5400, 1.2, 8, 20, 0, 2, 0, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 5) {
		vector<KeyPoint> keypoints;
		Ptr<FeatureDetector> detector = AgastFeatureDetector::create(5,true,3);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 6) {
		
		for (int i = 0; i < prev.rows-7; i = i+7) {
			for (int j = 0; j < prev.cols-7; j = j + 7) {
				points.push_back(Point2f(j, i));
			}
		}
	}
	
	// Lucas-Kanade point tracking
	cv::calcOpticalFlowPyrLK(prev_grayscale, cur_grayscale, points, dst_points, status, err, Size(11, 11), 4);
	
	// Set max vector length allowed (in pixels) N.B change max vector length for different data sets.
	int max_flow_length = 100;

	//long double execTime1 = (getTickCount()*1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	for (unsigned int i = 0; i < points.size(); i++)
	{
		if (status[i] != 0 && 
			sqrt(pow(points[i].x - dst_points[i].x, 2) + pow(points[i].y - dst_points[i].y, 2)) < max_flow_length &&
			dst_points[i].x >= 0 && dst_points[i].x < cur.cols && dst_points[i].y < cur.rows && dst_points[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(dst_points[i]);
		}
	}	
	
	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();
	
	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k);
	gd->setSigma(sigma);
	gd->setUsePostProcessing(use_post_proc);
	gd->setFGSLambda(fgs_lambda);
	gd->setFGSSigma(fgs_sigma);

	if (points_filtered.size() > SHRT_MAX) {
		cout << "Too many points to interpolate";
	}
	
	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);


	///////////////// 4. Variational refinement (optional - not used in final results as adds significant computation time) ///////////////

	// Split flow image into x and y components to pass to refinement
	//Mat U_V[2];   //destination array
	//split(dense_flow, U_V);//split source

	//int64 refinement_start = getTickCount();
	//// Source images to gray
	//Mat gray_1;
	//Mat gray_2;
	//cvtColor(from, gray_1, CV_RGB2GRAY);
	//cvtColor(to, gray_2, CV_RGB2GRAY);

	//int variational_refinement_iter = 3;
	//float variational_refinement_alpha = 20.f;
	//float variational_refinement_gamma = 10.f;
	//float variational_refinement_delta = 5.f;

	//Ptr<optflow::VariationalRefinement> variational_refinement_processor = optflow::createVariationalFlowRefinement();

	//variational_refinement_processor->setAlpha(variational_refinement_alpha);
	//variational_refinement_processor->setDelta(variational_refinement_delta);
	//variational_refinement_processor->setGamma(variational_refinement_gamma);
	//variational_refinement_processor->setSorIterations(5);
	//variational_refinement_processor->setFixedPointIterations(variational_refinement_iter);

	//variational_refinement_processor->calcUV(gray_1, gray_2, U_V[0], U_V[1]);

	//variational_refinement_processor->collectGarbage();

	//vector<Mat> ch;
	//flow.create(from.size(), CV_32FC2);
	//Mat dst = flow.getMat();
	//ch.push_back(U_V[0]);
	//ch.push_back(U_V[1]);
	//merge(ch, dst);
}




// DeGraF-Flow using Robust Local Optical Flow point tracking, requires RLOF code found at https://github.com/tsenst/RLOFLib
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/
void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));


	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	vector<Point2f> dst_points;
	vector<unsigned char> status;
	vector<float> err;
	int64 timeStart0 = getTickCount();
	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0) {
		cv::Size s = from.size();

		IplImage *dog_1 = cvCreateImage(cvSize(s.width, s.height), IPL_DEPTH_8U, 3);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(&(IplImage(from.getMat())), dog_1, 3, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();

		int status_1 = gradient_detector_1->DetectGradients(dog_1, 3, 3, 9, 9);  // DeGraF params specified here

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);
	}
	// Using other point detectors
	else if (point == 1) {
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints,10, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2) {
		vector<KeyPoint> keypoints;

		Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(0.1, 4, 2);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3) {
		vector<KeyPoint> keypoints;
		Ptr<FeatureDetector> detector = ORB::create(5400, 1.2, 8, 20, 0, 2, 0, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4) {
		vector<KeyPoint> keypoints;
		Ptr<FeatureDetector> detector = AgastFeatureDetector::create(16, true, 3);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 5) {
		for (int i = 0; i < prev.rows-6; i = i + 6) {
			for (int j = 0; j < prev.cols-6; j = j + 6) {
				points.push_back(Point2f(j, i));
			}
		}
	}
	else if (point == 6) {
		vector<KeyPoint> keypoints;
		Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(5400, 3, 0.00, 100.0);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}

	long double execTime0 = (getTickCount()*1.0000 - timeStart0) / (getTickFrequency() * 1.0000);
	std::cout << "Time to compute DeGraF points = " << execTime0 << "\n\n";

	//////////////////////////////// RLOF ////////////////////////////////////////////////////////////////

	int64 timeStart1 = getTickCount();

	rlof::Image img0, img1;
	std::vector<rlof::CRPoint> prevPoints, currPoints;
	img0.attach(prev);
	img1.attach(cur);

	for (int r = 0; r < points.size(); r++)
	{
		prevPoints.push_back(rlof::CRPoint(points[r].x, points[r].y));
	}

	// Change default RLOF parameters here
	rlof::Parameter rlof_Parmeter;
	rlof_Parmeter.m_UseIlluminationModel = true;
	rlof_Parmeter.m_UseGlobalMotionPrior = true;
	rlof_Parmeter.m_SmallWinSize = 10;
	rlof_Parmeter.m_LargeWinSize = 11;
	rlof_Parmeter.m_MaxLevel = 4;
	rlof_Parmeter.m_MaxIter = 30;
	rlof::SparseFlow * proc = rlof::SparseFlow::create(rlof_Parmeter);

	try
	{
		proc->run(img0, img1, prevPoints, currPoints);
	}
	catch (std::runtime_error & e)
	{
		std::cout << e.what() << std::endl;
	}

	// Clear memory by calling destructor
	proc->~SparseFlow();
	// clear points 
	dst_points.clear();
	dst_points_filtered.clear();
	points_filtered.clear();
	dst_points.shrink_to_fit();
	dst_points_filtered.shrink_to_fit();
	points_filtered.shrink_to_fit();
	
	// Convert rlof CRPoints back to Point2F
	for (int r = 0; r < currPoints.size(); r++)
	{
		dst_points.push_back(Point2f(currPoints[r].x, currPoints[r].y));
	}

	int max_flow_length = 100;
	for (unsigned int i = 0; i < points.size(); i++) {

		if (sqrt(pow(points[i].x - dst_points[i].x, 2) + pow(points[i].y - dst_points[i].y, 2)) < max_flow_length &&
			dst_points[i].x >= 0 && dst_points[i].x < cur.cols && dst_points[i].y < cur.rows && dst_points[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(dst_points[i]);
		}
	}

	long double execTime1 = (getTickCount()*1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	std::cout << "Time to run RLOF = " << execTime1 << "\n\n";


	////////////////////////////////   Interpolation  //////////////////////////////////////////////////////////////////

	int64 timeStart2 = getTickCount();

	if (points_filtered.size() > SHRT_MAX) {
		cout << "Too many points to interpolate";
	}

	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();

	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k);
	gd->setSigma(sigma);
	gd->setUsePostProcessing(use_post_proc);
	gd->setFGSLambda(fgs_lambda);
	gd->setFGSSigma(fgs_sigma);

	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);

	long double execTime2 = (getTickCount()*1.0000 - timeStart2) / (getTickFrequency() * 1.0000);
	std::cout << "Time to interpolate = " << execTime2 << "\n";
}