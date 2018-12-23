// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file PluginGradientFlow/GradientDetector.h
  \brief GradientDetector class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
#include <opencv\cv.h>

//FS new for descriptor fuicntion (use Mats)
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

//#include "VisionerLibTypes.h"

// Gradient structure
struct Gradient
{
    bool active, is_keypoint;
    int x, y;
    float dx, dy;
    CvPoint low_centroid, high_centroid;
    CvPoint2D32f centre, centroid;
    double magnitude, angle, pos_neg_ratio, xy_ratio, average, weight;
};


//! A class of image array functions
class GradientDetector {
private:
    // Private variables
    bool init_flag;
    CvSize image_size, window_size;
    int step_x, step_y;
    IplImage *image_8u, *image_f32;

public:
    // Public variables    
    Gradient **gradient_matrix;
    CvSize matrix_size;
    std::vector<cv::KeyPoint> keypoints;

	// FS addition
	cv::Mat descriptors;
	cv::Mat magnitudes;
	bool create_degraf_image = true;

    // Public functions
    GradientDetector();
    ~GradientDetector();
    void Create(IplImage* p_image_src, int p_window_width, int p_window_height, int p_step_x, int p_step_y);
    int DetectGradients(IplImage* p_image, int p_window_width = 2, int p_window_height = 2, int p_step_x = 1, int p_step_y = 1);
	
    void Release(void);
};

