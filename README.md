# How Big Is This?
An attempt at using OpenCV to find dimensions of various objects in the image. In the final commit, I take an object, with known dimensions, and using Canny edge detection, dilation and erosion, over the Greyscaled, Gaussian Blurred image, to get all the contours present in the image. I loop through the ordered contours, where the leftmost eligible contour(eligibility is suffient size/area of the contour) is the known object-it is positioned as such for this purpose. 

Using basic mathematics, for finding midpoint, etc, of the bounding box resulting from the contour, I get the dimensions of the contour. In the case of the first contour, i.e. the known object, this allows me to set a metric called pixelsPerMetric. This sets a scaling standard through which all the other contours' dimensions are set. 

Finally, all the midpoints, bounding boxes, and the dimensions are displayed in turn for each object in the image(that has some minimum considerable area).
