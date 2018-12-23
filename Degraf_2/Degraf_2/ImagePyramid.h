// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file ImagePyramid.h
  \brief ImagePyramid class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "windows.h"                  // new to allow TRUE and FALSE
//#include "VisionerLibTypes.h"
//#include "VisionerLibDefs.h"

using namespace std;

//! A class of pyramidal image functions
class ImagePyramid {
private:

protected:

public:
    // Public Variables
    int init_status;
    int image_depth, pyramid_height;
    int *level_scale;
    CvSize image_size, *level_size;
    IplImage **level_image; // list of pointers?

    // Constructor & Destructor
    ImagePyramid();
    ~ImagePyramid();

    // Public Function Prototypes
    void Create(IplImage* p_src, uint p_pyr_levels);
    int BuildPyramidUp(IplImage* p_src, int p_pyr_levels, double p_scale = 1, double p_shift = 0);
    int BuildPyramidDown(IplImage* p_src, double p_scale = 1, double p_shift = 0);
    void Release(void);
    int CheckImage(IplImage *p_image_src, int p_pyr_levels);
};

// Function Prototypes
void CopyImagePyramid(ImagePyramid *p_src_pyramid, ImagePyramid *p_dest_pyramid);
