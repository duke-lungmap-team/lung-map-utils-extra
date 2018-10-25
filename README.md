# OpenCV Extras
cv2-extras (cv2x) is a Python library for higher level functions used in image 
analysis and computer vision. OpenCV is an incredible library for building 
image analysis pipelines, but sometimes feels quite low-level, and it should 
stay that way. However, for those times when the lazy programmer finds 
themselves writing the same lines over and over or when there is just something
too useful to keep to yourself...there's cv2x.

Feel free to submit pull requests if some repetitive task is bugging you.

## Usage

Recommended import convention:

`import cv2_extras as cv2x`

Functions:

* fill_holes(mask)
* filter_contours_by_size(mask, min_size=1024, max_size=None)
* determine_channel_mode(channel)
* find_border_contours(contours, img_h, img_w)
* fill_border_contour(contour, img_shape)
* find_border_by_mask(signal_mask,
        contour_mask,
        max_dilate_percentage=0.2,
        dilate_iterations=1)
* find_contour_union(contour_list, img_shape)
* generate_background_contours(hsv_img,
        non_bg_contours,
        remove_border_contours=True,
        plot=False)
* elongate_contour(contour, img_shape, extend_length)