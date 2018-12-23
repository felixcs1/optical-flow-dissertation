// Header file for the modified version of Avi Singh's VO application found at:
// https://github.com/avisingh599/mono-vo


#pragma once

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "SaliencyDetector.h"
#include "GradientDetector.h"

#include <RLOF_Flow.h>

using namespace cv;
using namespace std;


class Odometry {
	public:
		Odometry();
		int run();
		int runGroundTruth();
};




