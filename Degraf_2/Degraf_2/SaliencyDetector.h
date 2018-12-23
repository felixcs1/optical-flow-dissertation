// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file SaliencyDetector.h
  \brief SaliencyDetector class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
//#include "VisionerLibDefs.h"
//#include "VisionerLibTypes.h"
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "ImagePyramid.h"
#include "ImageArray.h"

#include <iostream>		// standard C++ I/O FS for debugging
#include <string>	

//! A class for detecting visual saliency
class SaliencyDetector {
private:
    // Private variables
    int init_status;
    int image_depth;
    CvSize image_size;
    uint pyramid_height;    

protected:

public:
    // Public Variables
    ImagePyramid *pyramid, *pyramid_inv;
    ImageArray *image_3ch;
    IplImage *image_8u;
    IplImage *matrix_ratio, *matrix_ratio_inv, *matrix_min_ratio, *unit_matrix;
    IplImage *saliency_matrix;

    // Constructor & Destructor
    SaliencyDetector();
    ~SaliencyDetector();

    // Public Function Prototypes
    void Create(IplImage* p_image_src, uint p_pyr_levels);
    void Release(void);
    int DIVoG_Saliency(IplImage* p_image_src, IplImage* p_image_dest = NULL, int p_pyr_levels = 3, bool p_filter = false, bool p_norm = false);
    int DoGoS_Saliency(IplImage* p_image_src, IplImage* p_image_dest = NULL, int p_pyr_levels = 3, bool p_filter = false, bool p_norm = false);
    int CheckImage(IplImage *p_image_src, int p_pyr_levels);
};

