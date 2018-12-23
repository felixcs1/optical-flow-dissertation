// Modified version of Avi Singh's VO application found at:
// https://github.com/avisingh599/mono-vo
// Added DeGraF points and a function to draw the ground truth trajectory 

#include "stdafx.h"
#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 4539
#define MIN_NUM_FEAT 2000

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

Odometry::Odometry() {

}

vector<Mat> loadPoses(string file_name) {
	vector<Mat> poses;
	FILE *fp = fopen(file_name.c_str(), "r");
	if (!fp)
		return poses;

	
	while (!feof(fp)) {
		
		Mat P = Mat::eye(3, 4, CV_64FC1);
		if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
			&P.at<long float>(0,0), &P.at<long float>(0,1), &P.at<long float>(0,2), &P.at<long float>(0, 3),
			&P.at<long float>(1, 0), &P.at<long float>(1, 1), &P.at<long float>(1, 2), &P.at<long float>(1, 3),
			&P.at<long float>(2,0), &P.at<long float>(2, 1), &P.at<long float>(2,2), &P.at<long float>(2,3)) == 12) {		
			poses.push_back(P);
		}
	}
	fclose(fp);
	return poses;
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status) {
	
	//this function automatically gets rid of points for which tracking fails
	vector<float> err;
	Size winSize = Size(21, 21);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

	calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

	//getting rid of points for which the LK tracking failed or those who have gone outside the frame
	int indexCorrection = 0;
	for (int i = 0; i<status.size(); i++)
	{
		Point2f pt = points2.at(i - indexCorrection);
		if ((status.at(i) == 0) || (pt.x<0) || (pt.y<0)) {
			if ((pt.x<0) || (pt.y<0)) {
				status.at(i) = 0;
			}
			points1.erase(points1.begin() + (i - indexCorrection));
			points2.erase(points2.begin() + (i - indexCorrection));
			indexCorrection++;
		}

	}
}

void featureDetection(Mat img_1, vector<Point2f>& points1) { 
	// Use either FAST or DeGraF points 
	int point = 2;
	if (point == 1) {
		vector<KeyPoint> keypoints_1;
		int fast_threshold = 20;
		bool nonmaxSuppression = true;
		FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
		KeyPoint::convert(keypoints_1, points1, vector<int>());
		cout << points1.size() << "\n";
	}
	else if (point == 2) {

		// Clear memory
		points1.clear();
		points1.shrink_to_fit();

		cv::Size s = img_1.size();
		
		cvtColor(img_1, img_1, CV_GRAY2RGB);
		IplImage *dog_1 = cvCreateImage(cvSize(s.width, s.height), IPL_DEPTH_8U, 3);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(&(IplImage(img_1)), dog_1, 5, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();

		int status_1 = gradient_detector_1->DetectGradients(dog_1, 7, 7,5,5);


		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points1);
		
		// Release memory
		gradient_detector_1->Release();
		cvReleaseImage(&dog_1);
		img_1.release();
	}
}

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {

	string line;
	int i = 0;
	ifstream myfile("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/VO/00.txt"); // Change dir here
	double x = 0, y = 0, z = 0;
	double x_prev, y_prev, z_prev;
	if (myfile.is_open())
	{
		while ((getline(myfile, line)) && (i <= frame_id))
		{
			z_prev = z;
			x_prev = x;
			y_prev = y;
			std::istringstream in(line);
			for (int j = 0; j<12; j++) {
				in >> z;
				if (j == 7) y = z;
				if (j == 3)  x = z;
			}
			i++;
		}
		myfile.close();
	}

	else {
		cout << "Unable to open file";
		return 0;
	}

	return sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev));
}


int Odometry::run() {

	Mat img_1, img_2;
	Mat R_f, t_f; 

	//ofstream myfile;
	//myfile.open("results1_1.txt"); // file for printing numerical results to

	double scale = 1.00;
	char filename1[200];
	char filename2[200];
	sprintf(filename1, "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/VO/00/image_0/%06d.png", 0);  // Change dir here
	sprintf(filename2, "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/VO/00/image_0/%06d.png", 1);  // Change dir here

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	//read the first two frames from the dataset
	Mat img_1_c = imread(filename1);
	Mat img_2_c = imread(filename2);

	if (!img_1_c.data || !img_2_c.data) {
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	// we work with grayscale images
	cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

	// feature detection, tracking
	vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
	featureDetection(img_1, points1);        //detect features in img_1
	vector<uchar> status;
	featureTracking(img_1, img_2, points1, points2, status); //track those features to img_2

	double focal = 718.8560;
	cv::Point2d pp(607.1928, 185.2157);
	//recovering the pose and the essential matrix
	Mat E, R, t, mask;
	E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points2, points1, R, t, focal, pp, mask);


	cout << "R: " << R << "\nT: " << t;
	Mat prevImage = img_2;
	Mat currImage;
	vector<Point2f> prevFeatures = points2;
	vector<Point2f> currFeatures;

	char filename[100];

	R_f = R.clone();
	t_f = t.clone();

	clock_t begin = clock();

	namedWindow("Road facing camera", WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

	//Mat traj = Mat::zeros(600, 600, CV_8UC3);
	Mat traj = imread("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/images/output/VO/ground.png", 1);
	for (int numFrame = 0; numFrame < MAX_FRAME; numFrame++) {
		
		sprintf(filename, "C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/VO/00/image_0/%06d.png", numFrame);  // Change dir here

		Mat currImage_c = imread(filename);
		cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
		vector<uchar> status;
		featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
		
		E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

		Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

		for (int i = 0; i<prevFeatures.size(); i++) {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
			prevPts.at<double>(0, i) = prevFeatures.at(i).x;
			prevPts.at<double>(1, i) = prevFeatures.at(i).y;

			currPts.at<double>(0, i) = currFeatures.at(i).x;
			currPts.at<double>(1, i) = currFeatures.at(i).y;
		}

		scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
		
		if ((scale>0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

			t_f = t_f + scale*(R_f*t);
			R_f = R*R_f;
		}

		else {
			//cout << "scale below 0.1, or incorrect translation" << endl;
		}

		// lines for printing results
		// myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

		// a redetection is triggered in case the number of feautres being tracked go below a particular threshold
		if (prevFeatures.size() < MIN_NUM_FEAT) {
			featureDetection(prevImage, prevFeatures);
			featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
		}

		prevImage = currImage.clone();
		prevFeatures.clear();
		prevFeatures = currFeatures;
		currFeatures.clear(); // FS added to try to solve memory overflow when running on many frames

		int x = int(t_f.at<double>(0)) + 300;
		int y = int(t_f.at<double>(2)) + 100;
		circle(traj, Point(x, y), 1, CV_RGB(0, 0, 255), 1);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		imshow("Road facing camera", currImage_c);
		imshow("Trajectory", traj);

		waitKey(1);

	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	return 0;
}

// Added by FS to draw the groundtruth path
int Odometry::runGroundTruth() {
	cout << "Start";
	vector<Mat> poses = loadPoses("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/evaluation/VO/00.txt");  // Change dir here
	cout << "Poses loaded!";

	Mat R_f, t_f; 

	double scale = 1.00;

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	double focal = 718.8560;
	cv::Point2d pp(607.1928, 185.2157);
	//recovering the pose and the essential matrix
	Mat R, t, mask;

	R_f = poses[0].colRange(0,3);
	t_f = poses[0].col(3);

	cout << "T_f: " << t_f << "\n";
	cout << "R_f: " << R_f;

	clock_t begin = clock();

	namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

	Mat traj = Mat(700, 700, CV_8UC3, cv::Scalar(255,255,255));


	cout << "# POSES: " << poses.size();
	for (int row = 1; row < poses.size(); row++) {

		R_f = poses[row].colRange(0, 3);
		t_f = poses[row].col(3);
				
		scale = getAbsoluteScale(row+1, 0, t_f.at<double>(2)); // If error check this line for +1

		int x = int(t_f.at<double>(0)) + 300;
		int y = int(t_f.at<double>(2)) + 100;
		circle(traj, Point(x, y), 1, CV_RGB(0, 255, 0), 2);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		imshow("Trajectory", traj);

		waitKey(1);
	}

	imwrite("\a\folder\groundtruth.png", traj);  // Change dir here
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	return 0;
}