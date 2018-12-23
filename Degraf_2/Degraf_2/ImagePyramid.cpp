// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file ImagePyramid.cpp
  \brief ImagePyramid class implementation
  \author Ioannis Katramados
*/

// Include Files
#include "stdafx.h"
#include "ImagePyramid.h"

using namespace std;

//! Class constructor
ImagePyramid::ImagePyramid()
{
	const int DEFAULT_IMAGE_WIDTH = 640; //new
	const int DEFAULT_IMAGE_HEIGHT = 480; //new
    init_status = false;
    pyramid_height = 0;
    image_size.width = DEFAULT_IMAGE_WIDTH;
    image_size.height = DEFAULT_IMAGE_HEIGHT;
    image_depth = IPL_DEPTH_8U;
}

//! Class destructor
ImagePyramid::~ImagePyramid()
{
    Release();
}

//! Creates a pyramid of images
/*!
  \param p_image source image
  \param p_pyr_levels number of pyramid levels
  \todo malloc should be replaced with "new".
*/
void ImagePyramid::Create(IplImage* p_image, uint p_pyr_levels)
{
    // Local Variables
    int i;

    // Initialise variables
    pyramid_height = p_pyr_levels;

    // Allocate memory
    level_scale = (int*)malloc(sizeof(int)*pyramid_height);
    level_size = (CvSize*)malloc(sizeof(CvSize)*pyramid_height);
    level_image = (IplImage**)malloc(sizeof(IplImage*)*pyramid_height);

    //Initialise variables
    image_size = cvGetSize(p_image);
    level_size[0] = cvGetSize(p_image);
    level_image[0] = cvCreateImage(level_size[0], p_image->depth, p_image->nChannels); // Allocate memory for pyramid's base

    // Copy source image to the bottom of the pyramid
    cvCopy(p_image, level_image[0]);

    // Derive any subsequent pyramid level by resolution reduction
    level_scale[0] = 1;
    for (i = 1; i < pyramid_height; i++)
    {
        if(i > 0)
        {
            level_scale[i] = level_scale[i-1]*2;
        }
        level_size[i].width = level_size[i-1].width/2; // Set image width for the current pyramid level
        level_size[i].height = level_size[i-1].height/2;  // Set image height for the current pyramid level
        level_image[i] = cvCreateImage(level_size[i], p_image->depth, p_image->nChannels); // Allocate memory for the current pyramid level
        cvPyrDown(level_image[i-1], level_image[i]); // Perform pyramidal resolution reduction
    }

    init_status = TRUE;
}

//! Builds a pyramid bottom-up
/*!
  \param p_image source image
  \param p_pyr_levels pyramid height
  \param p_scale scaling factor that is applied to the source array elements
  \param p_shift value added to the scaled source array elements
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::BuildPyramidUp(IplImage* p_image, int p_pyr_levels, double p_scale, double p_shift)
{
    // Local Variables
    int i;

    if(CheckImage(p_image, p_pyr_levels))
    {
        if(init_status == TRUE)
        {
            // Copy source image to the bottom of the pyramid
            cvConvertScale(p_image, level_image[0], p_scale, p_shift);

            // Derive any subsequent pyramid level by resolution reduction
            for (i = 1; i < pyramid_height; i++)
            {
                cvPyrDown(level_image[i-1], level_image[i]);
            }
            return(TRUE);
        }
    }
    return(FALSE);
}

//! Builds a pyramid top-down
/*!
  \param p_image source image
  \param p_scale scaling factor that is applied to the source array elements
  \param p_shift value added to the scaled source array elements
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::BuildPyramidDown(IplImage* p_image, double p_scale, double p_shift)
{
    // Local Variables
    int i;

    if(init_status == TRUE)
    {
        // Copy source image to the bottom of the pyramid
        cvConvertScale(p_image, level_image[pyramid_height-1], p_scale, p_shift);

        // Derive any subsequent pyramid level by resolution reduction
        for (i = pyramid_height-1; i > 0; i--)
        {
            cvPyrUp(level_image[i], level_image[i-1]);
        }
        return(TRUE);
    }
    return(FALSE);
}

//! Releases memory of a pyramid
void ImagePyramid::Release(void)
{
    int i;

    // Release Memory
    if(init_status == TRUE)
    {
        init_status = FALSE;

        for (i = 0; i < pyramid_height; i++)
        {
            cvReleaseImage(&level_image[i]);
        }
        free(level_scale);
        free(level_size);
        free(level_image);
    }
}

//! Checks an image is valid for processing
/*!
  \param p_image_src source image
  \param p_pyr_levels pyramid height
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::CheckImage(IplImage *p_image_src, int p_pyr_levels)
{
    // Check input
    if(p_image_src == NULL)
    {
        return(FALSE);
    }

    // Detect changes to image properties
    if(init_status == TRUE)
    {
        if(p_image_src->width != image_size.width || p_image_src->height != image_size.height ||\
                p_image_src->depth != image_depth || p_pyr_levels != pyramid_height)
        {
            // Release memory
            Release();
        }
    }

    // Check initialisation status
    if(init_status == FALSE)
    {
        Create(p_image_src, p_pyr_levels);
    }

    return(TRUE);
}

//! Copies an image pyramid
/*!
  \param p_src_pyramid source pyramid
  \param p_dest_pyramid destination pyramid
*/
void CopyImagePyramid(ImagePyramid *p_src_pyramid, ImagePyramid *p_dest_pyramid)
{
    // Local Variables
    int i, pyramid_height;

    if(p_src_pyramid->init_status == TRUE)
    {
        // Initialise variables
        if(p_dest_pyramid->init_status == FALSE)
        {
            p_dest_pyramid->Create(p_src_pyramid->level_image[0], p_src_pyramid->pyramid_height);
        }

        // Set pyramid height as the shortest
        pyramid_height = min(p_src_pyramid->pyramid_height, p_dest_pyramid->pyramid_height);

        // Copy the image from source to destination for each pyramid level.
        for (i = 0; i < pyramid_height; i++)
        {
            cvCopy(p_src_pyramid->level_image[i], p_dest_pyramid->level_image[i]);
        }
    }
}

