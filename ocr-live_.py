import cv2
import pytesseract
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyshine as ps
from vidgear.gears import WriteGear
import opencv_hough_lines as lq
import imutils

font ='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
video = "/Users/adegallaix/Downloads/ocr-video/test_video_1_1.mp4"
out_video = "/Users/adegallaix/Downloads/ocr-video/ocr_video.mp4"


def high_pass_filter(image, blur,sigma=1.0):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (blur, blur), sigma)
    # Subtract the blurred image from the original
    high_pass = cv2.subtract(image, blurred)
    # Add the high-pass image back to the original
    sharpened = cv2.addWeighted(image, 1.0, high_pass, 1.0, 0)
    return sharpened
    
def drawHoughLines(image, lines,output):
    out = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output, out)
    return out

def maskingfunction(img:np.array, blackChanelThresh, whiteChannelThresh):
    _ , mask = cv2.threshold(img, blackChanelThresh, whiteChannelThresh, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inv_mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    background = np.zeros_like(img, dtype=np.uint8)
    background[:] = [0, 0, 0]  # Black Background
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    final_result = cv2.add(result, background)
    return final_result

def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction with the first point been closest to 0,0
    Assumption:
        There are exactly 4 points in the input and
        from a rectangle which is not very distorted
    """
    if len(pts) != 4:
        return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        # Top-left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] < center[1]))[0][0], :],
        # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] < center[1]))[0][0], :],
        # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] > center[1]))[0][0], :],
        # Bottom-Left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] > center[1]))[0][0], :]
    ]

    return np.array(cyclic_pts)


def drawCircle(frame:np.array, circle_centre:list, rad:float, color:list,thickness:int):
    cv2.circle(frame, circle_centre, rad, color, thickness,lineType=cv2.LINE_AA,)

if __name__ == "__main__":
    #main()

    img = cv2.imread("photo.jpg")
    #img = cv2.imread(back)
##
    masked = maskingfunction(img, 180,185)
    highpass_filter = cv2.cvtColor(high_pass_filter(masked,33), cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(highpass_filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,10)
    
    edge_detection = cv2.Canny(binary, 254, 255, apertureSize=3)

    polar_lines = cv2.HoughLines(edge_detection,1,np.pi/180,285)

    hough_lines = img.copy()

    houghLines=drawHoughLines(hough_lines,polar_lines,"hough_lines.jpg")

    binary = cv2.cvtColor(edge_detection, cv2.COLOR_GRAY2BGR)
    final_output = np.hstack((masked, img, binary,houghLines))

    cv2.imshow('Masking', final_output)
    cv2.waitKey(0)

    
    intersect_pts = lq.hough_lines_intersection(polar_lines, cv2.cvtColor(houghLines,cv2.COLOR_BGR2GRAY).shape)

    intersect_pts = cyclic_intersection_pts(intersect_pts)

    corners = img.copy()
    [drawCircle(img.copy(),intersect,6,(0,0,255),3) for intersect in intersect_pts]
    cv2.imshow('Corners',corners)
    cv2.waitKey(0)

    edge_detection = cv2.cvtColor(edge_detection, cv2.COLOR_GRAY2BGR)

    final_output = np.hstack((masked, img, edge_detection,hough_lines))
    cv2.imshow('Masking', final_output)
    cv2.imwrite("Scan.png", final_output)

    ##
    cv2.waitKey(0)
    '''
    
    #draw lines on undistorted image
    if polar_lines is not None:
        output = drawHoughLines(img,polar_lines)
        cv2.imshow('HoughLines', output)
        cv2.waitKey(0)
        intersection_pts = lq.hough_lines_intersection(polar_lines,binary.shape)
        intersection_pts = cyclic_intersection_pts(intersection_pts)
        print(intersection_pts)
        [drawCircle(img,intersect,6,(0,0,255),3) for intersect in intersection_pts]
        cv2.imshow('Corners',img)
        cv2.waitKey(0)

    cv2.imshow('Masking', np.hstack((masked, binary, img)))
##
    cv2.waitKey(0)
'''