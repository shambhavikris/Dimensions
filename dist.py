from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)

def order_points(pts):
    #sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:,0]), :]

    #grab the leftmost and rightmost points from the sorted x coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    #now sort the leftmost coordinates according to their y-coordinates so we can grab the topleft and bottom left points respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]),:]
    (tl,bl) = leftMost

    #now that we have the topleft coordinate use it as an anchor to calculate the Euclidean distance between the topleft and the rightmost points;
    #by the Pythagorean theorem the point with the largest distance will be the bottom right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    #return the coordinates in the topleft, topright, bottomright, bottomleft corner
    return np.array([tl, tr, br, bl], dtype="float32")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="path to the input image")
ap.add_argument("-w", "--width", type = float, required = True, help="width of the leftmost object in the image(in inches)")
args = vars(ap.parse_args())


#load, convert to greyscale, blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

#edge detection, then dilation + erosion to close gps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contours from left to right
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
refObj = None
pixelsPerMetric = None

#we will make our reference object the leftmost always

#loop over contours individually
for c in cnts:
    if cv2.contourArea(c) < 100:
        continue
    #contour not large enough, so ignore
    orig = image.copy()

    #compute the bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    #order the points of the contour such that they appear in the topleft, topright, bottomright, bottomleft order
    #then draw the outline of the rotated bounding box

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    #loop over the orignial points and draw them
    for(x,y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    #unpack the ordered bounding box, then compute the midpoint between the topleft and topright coordinates,
    #followed by the midpoint between bottomleft and bottomright coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    #compute the midpoints
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    #draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrY), int(trbrY)), 5, (255, 0, 0), -1)

    #draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

    #compute the Euclinean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))#height
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))#width

    #if the pixels per metric has not been initialised then compute it as the
    #ratio of pixels to supplied metric(inches)

    if pixelsPerMetric is None:
        pixelsperMetric = db / args["width"]

    #compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    #draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX-15), int(tltrY -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX+10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    #show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
