/*
 * aperture.h
 *
 *  Created on: Nov 23, 2016
 *      Author: amyznikov
 */

#pragma once

#ifndef __aperture_h__
#define __aperture_h__

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool extract_auto_aperture(const Mat & image, Mat * mask, int erode_size = 2, int border_size = 0, bool affected_by_dual_align = false);
void draw_aperture_border(Mat & image, int size, const Scalar & color = Scalar(0));
void draw_aperture_border(const Mat & src, Mat & dst, int size, const Scalar & color = Scalar(0));

#endif /* __create_aperture_mask_h__ */
