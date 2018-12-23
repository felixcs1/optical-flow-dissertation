// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file ImageArray.h
  \brief ImageArray class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
#include <opencv\cv.h>
//#include "VisionerLibTypes.h"

//! A class of image array functions
class ImageArray {
private:
    // Private variables
    CvSize image_size;

public:
    // Public variables
    bool init_flag;
    int array_length;
    IplImage **image;

    // Public functions
    ImageArray();
    ~ImageArray();
    void InitArray(IplImage* p_src, uint p_length);
    void ReleaseArray(void);
};

// Function Prototypes
void CopyImageArray(ImageArray *p_src_array, ImageArray *p_dest_array);
