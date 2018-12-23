// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file ImageArray.cpp
  \brief ImageArray class implementation
  \author Ioannis Katramados
*/

// Include Files
#include "stdafx.h"
#include "ImageArray.h"

//! Class constructor
ImageArray::ImageArray()
{
    init_flag = false;
}

//! Class destructor
ImageArray::~ImageArray()
{
}

//! Creates an image array
/*!
  \param p_image source image
  \param p_length number of image array elements
  \todo malloc should be replaced with "new".
*/
void ImageArray::InitArray(IplImage* p_image, uint p_length)
{
    // Local Variables
    int i;

    // Initialise variables
    array_length = p_length;

    // Allocate memory
    image = (IplImage**)malloc(sizeof(IplImage*)*array_length);

    //Initialise variables
    image_size = cvGetSize(p_image); // Get source image size

    // Derive any subsequent pyramid level by resolution reduction
    for (i = 0; i < array_length; i++)
    {
        image[i] = cvCreateImage(image_size, p_image->depth, p_image->nChannels); // Allocate memory for each image
        cvSetZero(image[i]);
    }

    init_flag = true;
}

//! Releases memory of an image array
void ImageArray::ReleaseArray(void)
{
    int i;

    // Release Memory
    if(init_flag == true)
    {
        for (i = 0; i < array_length; i++)
        {
            cvReleaseImage(&image[i]);
        }
        free(image);
    }
}

//! Copies an image array
/*!
  \param p_src_array source array
  \param p_dest_array destination array
*/
void CopyImageArray(ImageArray *p_src_array, ImageArray *p_dest_array)
{
    // Local Variables
    int i, array_length;

    if(p_src_array->init_flag == true)
    {
        // Initialise variables
        if(p_dest_array->init_flag == false)
        {
            p_dest_array->InitArray(p_src_array->image[0], p_src_array->array_length);
        }

        // Set pyramid height as the shortest
        array_length = std::min(p_src_array->array_length, p_dest_array->array_length);

        // Copy the image from source to destination for each pyramid level.
        for (i = 0; i < array_length; i++)
        {
            cvCopy(p_src_array->image[i], p_dest_array->image[i]);
        }
    }
}
