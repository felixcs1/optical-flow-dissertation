// Heavily modified version of the openCV optical_flow_evaluation.cpp file:
// https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_evaluation.cpp 
// Adapted to be able to evaluate KITTI and Middlebury datasets.

#include "stdafx.h"

#include "EvaluateOptFlow.h"
#include "FeatureMatcher.h"

using namespace std;
using namespace cv;
using namespace optflow;
using namespace std;


EvaluateOptFlow::EvaluateOptFlow(){

}

inline bool isFlowCorrect(const Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect(const Point3f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
		&& (fabs(u.z) < 1e9);
}

// Converts middlebury 2 channel flow image to 3 channel (CV_16UC3) KITTI format
/*!
\param flow 2 channel optical flow Mat, middlebury format  
\return 3 channel KITTI optical flow Mat
*/
static Mat convertToKittiFlow(const Mat_<Point2f>& flow) {

	Mat kittiFlow = cv::Mat::ones(flow.rows, flow.cols, CV_16UC3);// type 18
	
	int width = flow.cols;
	int height = flow.rows;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			
			const Point2f pt = flow(i, j);
			float u = pt.x;
			float v = pt.y;
			
				kittiFlow.at<Vec3s>(i, j)[1] = (uint16_t)max(min(v*64.0f + 32768.0f, 65535.0f), 0.0f);
				
				kittiFlow.at<Vec3s>(i, j)[2] = (uint16_t)max(min(u*64.0f + 32768.0f, 65535.0f), 0.0f);
			
		}
	}
	return kittiFlow;
}


// Converts KITTI ground truth image into a middlebury 2 channel image for evaluation
/*!
\param ground_truth_path string specifying the path to the KITTI 3 channel ground truth image
\return 2 channel Mat, ground truth optical flow 
*/
static Mat readKittiGroundTruth(String ground_truth_path) {

	String path = ground_truth_path;
	// NB opencv has order BGR => valid , v , u
	Mat image = imread(path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	Mat gt = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
	int width = image.cols;
	int height = image.rows;

	for (int32_t v = 0; v < height; v++) {
		for (int32_t u = 0; u < width; u++) {
			Vec3s val = image.at<Vec3s>(v, u);
			if (val[0] > 0) {
				Vec2f flow;
				if (val[2] > 0) {
					flow[0] = ((float)val[2] - 32768.0f) / 64.0f;
				}
				else {
					flow[0] = ((float)val[2] + 32768.0f) / 64.0f;
				}

				if (val[1] > 0) {
					flow[1] = ((float)val[1] - 32768.0f) / 64.0f;
				}
				else {
					flow[1] = ((float)val[1] + 32768.0f) / 64.0f;
				}
				gt.at<Vec2f>(v, u) = flow;
			}
			else {
				Vec2f flow;
				flow[0] = std::numeric_limits<float>::quiet_NaN();
				flow[1] = std::numeric_limits<float>::quiet_NaN();
				gt.at<Vec2f>(v, u) = flow;
			}
		}
	}
	return gt;
}

static Mat endpointError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2)
{
	Mat result(flow1.size(), CV_32FC1);

	for (int i = 0; i < flow1.rows; ++i)
	{
		for (int j = 0; j < flow1.cols; ++j)
		{
			const Point2f u1 = flow1(i, j);
			const Point2f u2 = flow2(i, j);

			if (isFlowCorrect(u1) && isFlowCorrect(u2))
			{
				const Point2f diff = u1 - u2;
				result.at<float>(i, j) = sqrt((float)diff.ddot(diff)); //distance				
			}
			else
				result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
		}
	}
	return result;
}

static Mat angularError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2)
{
	Mat result(flow1.size(), CV_32FC1);

	for (int i = 0; i < flow1.rows; ++i)
	{
		for (int j = 0; j < flow1.cols; ++j)
		{
			const Point2f u1_2d = flow1(i, j);
			const Point2f u2_2d = flow2(i, j);
			const Point3f u1(u1_2d.x, u1_2d.y, 1);
			const Point3f u2(u2_2d.x, u2_2d.y, 1);

			if (isFlowCorrect(u1) && isFlowCorrect(u2))
				result.at<float>(i, j) = acos((float)(u1.ddot(u2) / norm(u1) * norm(u2)));
			else
				result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
		}
	}
	return result;
}

// what fraction of pixels have errors higher than given threshold?
static float stat_RX(Mat errors, float threshold, Mat mask)
{
	CV_Assert(errors.size() == mask.size());
	CV_Assert(mask.depth() == CV_8U);

	int count = 0, all = 0;
	for (int i = 0; i < errors.rows; ++i)
	{
		for (int j = 0; j < errors.cols; ++j)
		{
			if (mask.at<char>(i, j) != 0)
			{
				++all;
				if (errors.at<float>(i, j) > threshold)
					++count;
			}
		}
	}
	return (float)count / all;
}

static float stat_AX(Mat hist, int cutoff_count, float max_value)
{
	int counter = 0;
	int bin = 0;
	int bin_count = hist.rows;
	while (bin < bin_count && counter < cutoff_count)
	{
		counter += (int)hist.at<float>(bin, 0);
		++bin;
	}
	return (float)bin / bin_count * max_value;
}

void EvaluateOptFlow::calculateStats(Mat errors, Mat mask = Mat(), bool display_images = false)
{
	float R_thresholds[] = { 0.5f, 1.f, 2.f, 3.f, 5.f, 10.f };
	float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
	if (mask.empty())
		mask = Mat::ones(errors.size(), CV_8U);
	CV_Assert(errors.size() == mask.size());
	CV_Assert(mask.depth() == CV_8U);

	//mean and std computation
	Scalar s_mean, s_std;
	float mean, std;
	meanStdDev(errors, s_mean, s_std, mask);
	mean = (float)s_mean[0];
	std = (float)s_std[0];
	printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

	// FS added to collect stats (printed out in main.cpp)
	stats_vector.push_back(mean); 
	stats_vector.push_back(std);

	//RX stats - displayed in percent
	float R;
	int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
	for (int i = 0; i < R_thresholds_count; ++i)
	{
		R = stat_RX(errors, R_thresholds[i], mask);
		printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
		stats_vector.push_back(R*100); // FS added to collect stats
	}

	//AX stats
	double max_value;
	minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

	Mat hist;
	const int n_images = 1;
	const int channels[] = { 0 };
	const int n_dimensions = 1;
	const int hist_bins[] = { 1024 };
	const float iranges[] = { 0, (float)max_value };
	const float* ranges[] = { iranges };
	const bool uniform = true;
	const bool accumulate = false;
	calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
		accumulate);
	int all_pixels = countNonZero(mask);
	int cutoff_count;
	float A;
	int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
	for (int i = 0; i < A_thresholds_count; ++i)
	{
		cutoff_count = (int)(floor(A_thresholds[i] * all_pixels + 0.5f));
		A = stat_AX(hist, cutoff_count, (float)max_value);
		//printf("A%.2f: %.2f\n", A_thresholds[i], A);
	}
}

static Mat flowToDisplay(const Mat flow)
{
	// Used to show gt for kitti data with NaN values
	Mat flow_copy = flow;
	for (int32_t v = 0; v < flow.rows; v++) {
		for (int32_t u = 0; u < flow.cols; u++) {
			if (cvIsNaN(flow.at<Vec2f>(v, u)[0])) {
				Vec2f f;
				f[0] = 0.0f;
				f[1] = 0.0f;
				flow_copy.at<Vec2f>(v, u) = f;
			}
		}
	}

	Mat flow_split[2];
	Mat magnitude, angle;
	Mat hsv_split[3], hsv, rgb;
	split(flow_copy, flow_split);

	///////////////////-- New colour display/////////////////////////
	Mat flowHSV, flowRGB;
	vector<cv::Mat> flowVec(3);

	cv::cartToPolar(flow_split[0], flow_split[1], flowVec[1], flowVec[0], true);
	flowVec[2] = cv::Mat::ones(flowVec[0].size(), flowVec[0].type()) * 255;
	cv::threshold(flowVec[1], flowVec[1], 4, 4, CV_THRESH_TRUNC);
	flowVec[1] = flowVec[1] * 0.25f;
	cv::merge(flowVec, flowHSV);
	cv::cvtColor(flowHSV, flowRGB, cv::COLOR_HSV2BGR);
	flowRGB.convertTo(flowRGB, CV_8UC3);
	return flowRGB;
}

static Mat flowToDisplay2(const Mat flow)
{
	Mat flow_split[2];
	Mat magnitude, angle;
	Mat hsv_split[3], hsv, rgb;
	split(flow, flow_split);
	cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
	normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
	hsv_split[0] = angle; // already in degrees - no normalization needed
	hsv_split[1] = Mat::ones(angle.size(), angle.type());
	hsv_split[2] = magnitude;
	merge(hsv_split, 3, hsv);
	cvtColor(hsv, rgb, COLOR_HSV2BGR);
	return rgb;
}


// Converts an error flow map into a linear heatmap blue -> red
// N.B set range of colour map in the function where specified
/*!
\param flow 2 channel mat of End Point Error vectors
\param mask Mask specifiying the pixels at which flow was recovered
\return error heat map from blue (low error) to red (high error)
*/
static Mat errorHeatMap(const Mat_<Point2f>& flow, Mat mask) {
	
	Mat flow_magnitudes(flow.size(), CV_32FC1);
	Mat mask_copy;
	mask.copyTo(mask_copy);

	for (int i = 0; i < flow.rows; i++) {
		for (int j = 0; j < flow.cols; j++) {
			
			if (mask_copy.at<char>(i, j) != 0) {
				float flow_mag = sqrt(flow(i, j).x*flow(i, j).x + flow(i, j).y*flow(i, j).y);

				// Set max error to adjust the colour mapping 
				if (flow_mag > 3) {
					flow_magnitudes.at<float>(i, j) = 3; // Adjust this range depending on the error range you want to see.
				}
				else {
					flow_magnitudes.at<float>(i, j) = flow_mag;
				}
			}
			else {
				flow_magnitudes.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}

	// Convert to 8 bit
	flow_magnitudes.convertTo(flow_magnitudes, CV_8U);

	Mat norm;
	// Normailse to get contrast
	normalize(flow_magnitudes, norm, 255, 0, NORM_INF);
	applyColorMap(norm, norm, COLORMAP_JET);
	bitwise_not(mask_copy, mask_copy);
	norm.setTo(Scalar(0, 0, 0), mask_copy);
	// Show the result:
	return norm;
}

// Runs an evalution for a given optical flow method 
// N.B need to take care in specifying Middlebury/KITTI file locations as well as data_set tag within the funciton 
/*!
\param method a string corresponding to an optical flow method 
\param display bool to specify if output images should be shown
\param image_no specifies the KITTI or middlebury image number 
\return error heat map from blue (low error) to red (high error)
*/
int EvaluateOptFlow::runEvaluation(String method, bool display_images, int image_no)
{
	//////////////////// **** CHANGE THE IMAGE PAIR AND GROUND TRUTH FILE LOCATIONs HERE **** ////////////////////////////////
	String data_set = "kitti"; // or "middlebury";

	// Convert image_num into string 
	String num = to_string(image_no);
	if (image_no < 10) {
		num = "00" + num;
	}
	else if (image_no < 100) {
		num = "0" + num;
	}

	num = "006"; // set to evaluate just a single image pair

	// Middlebury Image names 
	vector<String> image_names = { {"Venus"}, { "RubberWhale" }, {"Grove2"}, {"Grove3"}, { "Urban2" }, { "Urban3" }, {"Hydrangea"} };

	vector<String> image_names_eval = { { "Army" },{ "Backyard" },{ "Basketball" },{ "Dumptruck" },{ "Evergreen" },{ "Grove" } , \
										{ "Mequon" }, { "Schefflera" }, { "Teddy" }, { "Urban" }, { "Wooden" }, { "Yosemite" } };

	// Middlebury path names (indexed by image_num)
	//String i1_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/Middlebury/other-data/" + image_names[image_no] + "/frame10.png";
	//String i2_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/Middlebury/other-data/" + image_names[image_no] + "/frame11.png";
	//String groundtruth_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/Middlebury/other-gt-flow/" + image_names[image_no] + "/flow10.flo"; 

	// KITTI 2015 train (indexed by image_num)
	/*String i1_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_scene_flow/training/image_2/000" + num + "_10.png";
	String i2_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_scene_flow/training/image_2/000" + num + "_11.png";
	String groundtruth_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_scene_flow/training/flow_noc/000" + num + "_10.png";*/

	//// KITTI 2012 train (indexed by image_num)
	String i1_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_stereo_flow/training/colored_0/000" + num + "_10.png";
	String i2_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_stereo_flow/training/colored_0/000" + num + "_11.png";
	String groundtruth_path = "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/data_stereo_flow/training/flow_noc/000" + num + "_10.png";

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	String error_measure = "endpoint";
	String region = "all";
	
	// Initialise vectors of points to display the sparse vector field if display)images is true 
	// and either degraf_flow method is being used
	vector<Point2f> points1;
	vector<Point2f> points2;

	stats_vector.push_back(image_no);
	Mat im1, im2; // to keeep for display
	Mat i1, i2;
	Mat_<Point2f> flow, ground_truth;
	Mat computed_errors;
	i1 = imread(i1_path, 1);
	i2 = imread(i2_path, 1);
	im1 = i1;
	im2 = i2;

	if (!i1.data || !i2.data || i1.empty() || i2.empty())
	{
		printf("No image data \n");
		return -1;
	}
	if (i1.size() != i2.size() || i1.channels() != i2.channels())
	{
		printf("Dimension mismatch between input images\n");
		return -1;
	}
	// 8-bit images expected by all algorithms
	if (i1.depth() != CV_8U)
		cout << "convert";
		i1.convertTo(i1, CV_8U);
	if (i2.depth() != CV_8U)
		i2.convertTo(i2, CV_8U);

	if ((method == "farneback" || method == "tvl1" || method == "deepflow" || method == "DISflow_ultrafast" || method == "DISflow_fast" || method == "DISflow_medium") && i1.channels() == 3)
	{   // 1-channel images are expected
		cvtColor(i1, i1, COLOR_BGR2GRAY);
		cvtColor(i2, i2, COLOR_BGR2GRAY);
	}
	else if (method == "simpleflow" && i1.channels() == 1)
	{   // 3-channel images expected
		cvtColor(i1, i1, COLOR_GRAY2BGR);
		cvtColor(i2, i2, COLOR_GRAY2BGR);
	}

	//flow = Mat(i1.size[0], i1.size[1], CV_32FC2);
	//flow = Mat::zeros(i1.size(), CV_32FC2);
	Ptr<DenseOpticalFlow> algorithm;

	if (method == "farneback")
		algorithm = createOptFlow_Farneback();
	else if (method == "simpleflow")
		algorithm = createOptFlow_SimpleFlow();
	else if (method == "tvl1")
		algorithm = createOptFlow_DualTVL1();
	else if (method == "deepflow")
		algorithm = createOptFlow_DeepFlow();
	else if (method == "sparsetodenseflow")
		algorithm = createOptFlow_SparseToDense();
	else if (method == "pcaflow") {
		//FS removed prior option
		algorithm = createOptFlow_PCAFlow();
	}
	else if (method == "DISflow_ultrafast")
		algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_ULTRAFAST);
		//cv::Ptr<cv::optflow::DISOpticalFlow> algorithm = cv::optflow::createOptFlow_DIS(0);
	else if (method == "DISflow_fast")
		algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_FAST);
	else if (method == "DISflow_medium")
		algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_MEDIUM);
	else if (method == "degraf_flow_lk") {
	}
	else if (method == "degraf_flow_rlof") {
	}
	else
	{
		printf("Wrong method!\n");
		return -1;
	}

	double startTick, time;
	startTick = (double)getTickCount(); // measure time

	if (method == "degraf_flow_lk") {
		FeatureMatcher algorithm = FeatureMatcher();
		algorithm.degraf_flow_LK(i1, i2, flow, 60, (0.05000000075F), true, (500.0F), (1.5F));

		if (display_images) {
			// Points for displaying sparse flow field
			points1 = algorithm.points_filtered;
			points2 = algorithm.dst_points_filtered;
		}
	}
	else if (method == "degraf_flow_rlof") {
		FeatureMatcher algorithm = FeatureMatcher();
		algorithm.degraf_flow_RLOF(i1, i2, flow, 127, (0.05000000075F), true, (500.0F), (1.5F));
		
		if (display_images) {
			// Points for displaying sparse flow field
			points1 = algorithm.points_filtered;
			points2 = algorithm.dst_points_filtered;
		}
	}
	else {
		algorithm->calc(i1, i2, flow);
	}

	time = ((double)getTickCount() - startTick) / getTickFrequency();
	printf("\nTime [s]: %.3f\n", time);

	if (!groundtruth_path.empty())
	{ // compare to ground truth
		if (data_set == "middlebury") {
			ground_truth = optflow::readOpticalFlow(groundtruth_path); // Middlebury 
		}
		else {
			ground_truth = readKittiGroundTruth(groundtruth_path); // KITTI
		}

		if (flow.size() != ground_truth.size() || flow.channels() != 2
			|| ground_truth.channels() != 2)
		{
			printf("Dimension mismatch between the computed flow and the provided ground truth\n");
			return -1;
		}
		if (error_measure == "endpoint")
			computed_errors = endpointError(flow, ground_truth);
		else if (error_measure == "angular")
			computed_errors = angularError(flow, ground_truth);
		else
		{
			printf("Invalid error measure! Available options: endpoint, angular\n");
			return -1;
		}

		Mat mask;
		if (region == "all")
			mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
		else if (region == "discontinuities")
		{
			Mat truth_merged, grad_x, grad_y, gradient;
			vector<Mat> truth_split;
			split(ground_truth, truth_split);
			truth_merged = truth_split[0] + truth_split[1];

			Sobel(truth_merged, grad_x, CV_16S, 1, 0, -1, 1, 0, BORDER_REPLICATE);
			grad_x = abs(grad_x);
			Sobel(truth_merged, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_REPLICATE);
			grad_y = abs(grad_y);
			addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!

			Scalar s_mean;
			s_mean = mean(gradient);
			double threshold = s_mean[0]; // threshold value arbitrary
			mask = gradient > threshold;
			dilate(mask, mask, Mat::ones(9, 9, CV_8U));
		}
		else if (region == "untextured")
		{
			Mat i1_grayscale, grad_x, grad_y, gradient;
			if (i1.channels() == 3)
				cvtColor(i1, i1_grayscale, COLOR_BGR2GRAY);
			else
				i1_grayscale = i1;
			Sobel(i1_grayscale, grad_x, CV_16S, 1, 0, 7);
			grad_x = abs(grad_x);
			Sobel(i1_grayscale, grad_y, CV_16S, 0, 1, 7);
			grad_y = abs(grad_y);
			addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!
			GaussianBlur(gradient, gradient, Size(5, 5), 1, 1);

			Scalar s_mean;
			s_mean = mean(gradient);
			// arbitrary threshold value used - could be determined statistically from the image?
			double threshold = 1000;
			mask = gradient < threshold;
			dilate(mask, mask, Mat::ones(3, 3, CV_8U));
		}

		else
		{
			printf("Invalid region selected! Available options: all, discontinuities, untextured");
			return -1;
		}

		//masking out NaNs and incorrect GT values
		Mat truth_split[2];
		split(ground_truth, truth_split);
		Mat abs_mask = Mat((abs(truth_split[0]) < 1e9) & (abs(truth_split[1]) < 1e9));
		Mat nan_mask = Mat((truth_split[0] == truth_split[0]) & (truth_split[1] == truth_split[1]));
		bitwise_and(abs_mask, nan_mask, nan_mask);
		bitwise_and(nan_mask, mask, mask); //including the selected region
		
		if (display_images)
		{
			Mat difference = ground_truth - flow;
			Mat masked_difference;
			difference.copyTo(masked_difference, mask);
			Mat error = flowToDisplay(masked_difference);
			Mat heatmap = errorHeatMap(difference, mask);
			Mat ground = flowToDisplay(ground_truth);
			Mat flow_image = flowToDisplay(flow);

			// Display all useful output images in one frame
			cv::Mat win_mat(cv::Size(i1.cols*2, i1.rows*3), CV_8UC3);


			imwrite("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/Images/output/Presentation/06_0.png", i1);
			imwrite("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/Images/output/Presentation/06_1.png", i2);
			// Draw sparse flow field from degraf flow 
			if (method == "degraf_flow_lk" || method == "degraf_flow_rlof") {
				Mat sparse = Mat::zeros(i1.rows, i1.cols, CV_8UC3);
				bitwise_not(sparse, sparse);
				for (int i = 0; i < points1.size(); i += 4)
				{
					cv::arrowedLine(sparse, points1[i], points2[i], cv::Scalar(0, 0, 0), 2, 8, 0, 0.2);
				}
				//imwrite("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/Images/output/Presentation/lines.png" , sparse);
				// place in output window
				sparse.copyTo(win_mat(cv::Rect(0, i1.rows, i1.cols, i1.rows)));
			}
			else {
				Mat placeholder = Mat::zeros(i1.rows, i1.cols, CV_8UC3);
				bitwise_not(placeholder, placeholder);
				cv::line(placeholder, Point2f(0,0), Point2f(i1.cols, i1.rows), cv::Scalar(0, 0, 0), 2, 8);
				cv::line(placeholder, Point2f(i1.cols, 0), Point2f(0, i1.rows), cv::Scalar(0, 0, 0), 2, 8);
				placeholder.copyTo(win_mat(cv::Rect(0, i1.rows, i1.cols, i1.rows)));
			}
			imwrite("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/Images/output/Presentation/flow.png", flow_image);
			im1.copyTo(win_mat(cv::Rect(0, 0, i1.cols, i1.rows)));
			im2.copyTo(win_mat(cv::Rect(i1.cols, 0, i1.cols, i1.rows)));
			flow_image.copyTo(win_mat(cv::Rect(i1.cols, i1.rows, i1.cols, i1.rows)));
			ground.copyTo(win_mat(cv::Rect(0, i1.rows*2, i1.cols, i1.rows)));
			heatmap.copyTo(win_mat(cv::Rect(i1.cols, i1.rows*2, i1.cols, i1.rows)));

			// Shrink to fit all image on the screen 
			resize(win_mat, win_mat, Size(1325, 600));
			imshow("Results", win_mat);
		}
		printf("Using %s error measure\n", error_measure.c_str());
		calculateStats(computed_errors, mask, display_images);

		if(display_images)
			waitKey(0);
	}
	if (display_images) // wait for the user to see all the images
		waitKey(1);

	// Collect stats from evaluation 
	stats_vector.push_back(time);
	all_stats.push_back(stats_vector);
	stats_vector.clear();

	return 0;
}

