// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file SaliencyDetector.cpp
  \brief SaliencyDetector class implementation
  \author Ioannis Katramados
*/

// Include Files
#include "stdafx.h"
#include "SaliencyDetector.h"

//#include <iostream>	// new
//
//using namespace std;

//! Class constructor
SaliencyDetector::SaliencyDetector()
{
    // Initialise variables
	const int DEFAULT_IMAGE_WIDTH = 640; //new
	const int DEFAULT_IMAGE_HEIGHT = 480; //new
    init_status = FALSE;
    image_size.width = DEFAULT_IMAGE_WIDTH;
    image_size.height = DEFAULT_IMAGE_HEIGHT;
    image_depth = IPL_DEPTH_8U;
    pyramid_height = 3;
}

//! Class destructor
SaliencyDetector::~SaliencyDetector()
{
}

//! Initialises a saliency detector
/*!
  \param p_image source image
  \param p_pyr_levels number of pyramid levels
*/
void SaliencyDetector::Create(IplImage* p_image, uint p_pyr_levels)
{
    // Local Variables
    IplImage *image_32f;

    // Get source image dimensions
    image_size = cvGetSize(p_image);
    image_depth = p_image->depth;
    pyramid_height = p_pyr_levels;

    // Setup image templates
    image_32f = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    image_8u = cvCreateImage(image_size, IPL_DEPTH_8U, 1);

    // Initialise pyramids
    pyramid = new ImagePyramid();
    pyramid->Create(image_32f, pyramid_height);
    pyramid_inv = new ImagePyramid();
    pyramid_inv->Create(image_32f, pyramid_height);
    cvReleaseImage(&image_32f);

    // Initialise image arrays
    image_32f = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    image_3ch = new ImageArray();
    image_3ch->InitArray(image_32f, 3);
    cvReleaseImage(&image_32f);

    // Initialise images
    saliency_matrix = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    matrix_ratio = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    matrix_ratio_inv = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    matrix_min_ratio = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    unit_matrix = cvCreateImage(image_size, IPL_DEPTH_32F, 1);
    cvSet(unit_matrix, cvScalar(1.0, 1.0, 1.0));

    // Set initialisation flag
    init_status = TRUE;
}

//! Releases saliency detector
void SaliencyDetector::Release(void)
{
    // Free memory
    cvReleaseImage(&image_8u);
    cvReleaseImage(&saliency_matrix);
    cvReleaseImage(&matrix_ratio);
    cvReleaseImage(&matrix_ratio_inv);
    cvReleaseImage(&matrix_min_ratio);
    cvReleaseImage(&unit_matrix);
    pyramid->Release();
    pyramid_inv->Release();
    image_3ch->ReleaseArray();

    // Reset initialisation flag
    init_status = FALSE;
}

//! Calculates per-pixel visual saliency using Division of Gaussians (DIVoG)
/*!
  \param p_image_src source image
  \param p_image_dest destination image
  \param p_pyr_levels pyramid height
  \param p_filter filter activation flag
  \param p_norm normalisation flag
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::DIVoG_Saliency(IplImage* p_image_src, IplImage* p_image_dest, int p_pyr_levels, bool p_filter, bool p_norm)
{
    // Local Variables
    CvScalar avg;

    // Check input
    if(CheckImage(p_image_src, p_pyr_levels))
    {
        // Convert to grayscale
        cvCvtColor(p_image_src, image_8u, CV_RGB2GRAY);

        // Create a pyramid of resolutions. Shift image by 2^n to avoid division
        // by zero or any number in the range 0.0 - 1.0;
        pyramid->BuildPyramidUp(image_8u, p_pyr_levels, 1.0, 2^pyramid_height);
        pyramid_inv->BuildPyramidDown(pyramid->level_image[pyramid_height-1]);

        // Calculate Minimum Ratio (MiR) matrix
        cvDiv(pyramid->level_image[0], pyramid_inv->level_image[0], matrix_ratio);
        cvDiv(pyramid_inv->level_image[0], pyramid->level_image[0], matrix_ratio_inv);
        cvMin(matrix_ratio, matrix_ratio_inv, matrix_min_ratio);;

        // Derive salience by subtracting from unit matrix
        cvSub(unit_matrix, matrix_min_ratio, saliency_matrix);
        cvConvertScale(saliency_matrix, image_8u, 255.0);

        // Low-pass filter
        if(p_filter)
        {
            avg = cvAvg(image_8u);
            cvSubS(image_8u, avg, image_8u);
        }

        // Normalization to range 0-255
        if(p_norm)
        {
            cvNormalize(image_8u, image_8u, 0, 255, CV_MINMAX);
        }

        // Generate output if a destination image is given as function parameter
        if(p_image_dest != NULL)
        {
            if(p_image_src->nChannels == 1) // Process grayscale saliency matrix
            {
                cvCopy(image_8u, p_image_dest);
            }
            else // Process colour saliency matrix
            {
                // Convert to colour
                cvCvtColor(image_8u, p_image_dest, CV_GRAY2RGB);
            }            
        }
        return(TRUE);
    }
    return(FALSE);
}

//! Calculates per-pixel visual saliency using Difference of Gaussians (DoGoS)
/*!
  \param p_image_src source image
  \param p_image_dest destination image
  \param p_pyr_levels pyramid height
  \param p_filter filter activation flag
  \param p_norm normalisation flag
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::DoGoS_Saliency(IplImage* p_image_src, IplImage* p_image_dest, int p_pyr_levels, bool p_filter, bool p_norm)
{
    // Local Variables
    CvScalar avg;

    // Check input
    if(CheckImage(p_image_src, p_pyr_levels))
    {
        // Convert to grayscale
		cvCvtColor(p_image_src, image_8u, CV_RGB2GRAY);
		
        // Create a pyramid of resolutions. Shift image by 2^n to avoid division
        // by zero or any number in the range 0.0 - 1.0;
        pyramid->BuildPyramidUp(image_8u, p_pyr_levels, 1.0, 1.0);
        pyramid_inv->BuildPyramidDown(pyramid->level_image[pyramid_height-1]);
		
        // Calculate Minimum Ratio (MiR) matrix
        cvAbsDiff(pyramid->level_image[0], pyramid_inv->level_image[0], matrix_ratio);
        cvAdd(pyramid_inv->level_image[0], pyramid->level_image[0], matrix_ratio_inv);
        cvDiv(matrix_ratio, matrix_ratio_inv, saliency_matrix);
        cvConvertScale(saliency_matrix, image_8u, 255.0);

        // Low-pass filter
        if(p_filter)
        {
            avg = cvAvg(image_8u);
            cvSubS(image_8u, avg, image_8u);
        }

        // Normalization to range 0-255
        if(p_norm)
        {
            cvNormalize(image_8u, image_8u, 0, 255, CV_MINMAX);
        }

        // Generate output if a destination image is given as function parameter
        if(p_image_dest != NULL)
        {
            if(p_image_src->nChannels == 1) // Process grayscale saliency matrix
            {
                cvCopy(image_8u, p_image_dest);
            }
            else // Process colour saliency matrix
            {
                // Convert to colour
                cvCvtColor(image_8u, p_image_dest, CV_GRAY2RGB);
            }
        }
        return(TRUE);
    }
    return(FALSE);
}

//! Checks an image is valid for processing
/*!
  \param p_image_src source image
  \param p_pyr_levels pyramid height
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::CheckImage(IplImage *p_image_src, int p_pyr_levels)
{
    // Check input
    if(p_image_src == NULL)
    {
        return(FALSE);
    }

    // Detect changes to image properties
    if(init_status == TRUE)
    {
        if(p_image_src->width != image_size.width || p_image_src->height != image_size.height\
                || p_image_src->depth != image_depth || p_pyr_levels != pyramid_height)
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
