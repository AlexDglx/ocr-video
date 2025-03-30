import cv2
import pytesseract
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyshine as ps
from vidgear.gears import WriteGear
import opencv_hough_lines as lq
from averaged_pairs import average_close_pairs
import json


font ='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
video = "/Users/adegallaix/Downloads/ocr-video/rearBook.MOV"
out_video = "/Users/adegallaix/Downloads/ocr-video/ocr_video.mp4"
calib_path = "Apple_iPhone 12 Pro Max_1x__4k_16by9_3840x2160-30.00fps.json"

def readJson(jsonPath: str) -> dict:
    """Read JSON camera configuration file.
    param p1: describe about parameter p1
    jsonPath: str
        Document string 
    """
    with open(jsonPath, "r") as file:
        a = json.load(file)
        file.close()
    return a

def main():
    cap = cv2.VideoCapture(video)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_params =  {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-r":f'{frame_rate}',"-pix_fmt":"yuv420p", "-vf":f'setpts= {0.78}*PTS'}
    writer = WriteGear(output=out_video, compression_mode=True, logging=True, **output_params)

    while True:
        ret, img = cap.read()
        if not ret :
            print("Image not streamed")
            break
        # image text detection
        scale = 0.5
        font_scale = 1.5
        img = cv2.resize(img, (0, 0), fx = scale, fy = scale) #Resizing half
        original = img.copy()
        img_undistorted = resizeUndistort(img, calib_path)

        img_scan = resolveHomography(img_undistorted, 160 , 255,180)
        masked = maskingfunction(img_scan, 180,255)
        highpass_filter = cv2.cvtColor(high_pass_filter(masked,1), cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(highpass_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,10)
        #o  = pytesseract.image_to_osd()
        try:
            d = pytesseract.image_to_data(binary,config='--oem 3 --psm 1 -l fra ',output_type=pytesseract.Output.DICT)
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                if int(d['conf'][i]) > 0:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    #cv2.rectangle(resized, (x, y), (x + w, y + h), (75,21,121), -1)
                    #cv2.putText(resized, f'{d["text"][i]}',(x,y+25),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),1, cv2.LINE_AA)
                    try:
                        ps.putBText(img_scan,f'{d["text"][i]}',text_offset_x=x,text_offset_y=y,vspace=2,hspace=1, font_scale=3,background_RGB=(16, 58, 124),text_RGB=(255,255,255),font=1,thickness=2,alpha=0.25,gamma=0.5)
                    except Exception:
                        pass
        except:
            pass
                
                    #ps.putBText(resized,f'No Text',text_offset_x=480,text_offset_y=280,vspace=0,hspace=0, font_scale=1.25,background_RGB=(0,250,250),text_RGB=(255,250,250))
        #cv2.putText(img, detected_text, (50,50), 2, 1, (255,0,0), 1)
        #cv2.putText(resized_gray, detected_text, (50,50), 2, 1, (255,0,0), 1)
        img_scan = matchHeight(img_scan, original)
        #hstack_ = np.hstack((original,img_scan))
        cv2.imshow("Text Reader",img_scan)
        resized_image1 = cv2.resize(img_scan, (1080, 1080),interpolation = cv2.INTER_LINEAR)
        writer.write(resized_image1)

        #except Exception:
#
        #    img_undistorted = matchHeight(img_undistorted, original)
        #    ps.putBText(img_undistorted,f'{"No Data"}',text_offset_x=50,text_offset_y=50,vspace=2,hspace=1, font_scale=3,background_RGB=(199, 49, 49),text_RGB=(255,255,255),font=1,thickness=2,alpha=0.5,gamma=0.5)
        #    hstack_ =  np.hstack((original,img_undistorted))
        #    cv2.imshow("Text Reader",hstack_)
        #    #writer.write(img_undistorted)
        #    pass

        k = cv2.waitKey(1)
        if k%256 == ord('q'):
            print('Escape hit, closing image capture...')
            break

        elif k%256 == 32:
            print("Taking a picture") 
            cv2.imwrite("capture.png", img)
            cv2.imwrite("processed-image.png", binary)
    print("Video Processed")

    cap.release()
    writer.close()
    cv2.destroyAllWindows()
    

def high_pass_filter(image, blur,sigma=1.0):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (blur, blur), sigma)
    # Subtract the blurred image from the original
    high_pass = cv2.subtract(image, blurred)
    # Add the high-pass image back to the original
    sharpened = cv2.addWeighted(image, 1.0, high_pass, 1.0, 0)
    return sharpened
    
def drawHoughLines(image, lines):

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
        cv2.line(out, (x1, y1), (x2, y2), (252, 255, 3), 2)

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
    if pts.shape[0] != 4:
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

def matchHeight(image1,image2):
    # Get the height of the second image (target image)
    height2 = image2.shape[0]

    # Calculate the scaling factor to match the height of image2
    height1, width1 = image1.shape[:2]
    scaling_factor = height2 / float(height1)

    # Calculate the new width based on the scaling factor
    new_width = int(width1 * scaling_factor)

    # Resize the first image to match the height of the second image
    resized_image1 = cv2.resize(image1, (new_width, height2))

    return resized_image1

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3

def resizeUndistort(img:np.ndarray,calibrationFilePath):
    # resize file to match calibration dimensions
    cameraData = readJson(calibrationFilePath)
    if cameraData is None:
        raise FileNotFoundError(f"Image at path '{calibrationFilePath}' could not be loaded.")
    w,h = cameraData["calib_dimension"]['h'],cameraData["calib_dimension"]['w']
    output = cv2.resize(img, (w, h),interpolation = cv2.INTER_LINEAR)
    root = cameraData['fisheye_params']
    camMatrix, distortion = root['camera_matrix'], root['distortion_coeffs']


    #newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(np.array(camMatrix), np.array(distortion),(h,w),1,(h,w))

    undistorted_img = cv2.fisheye.undistortImage(output, np.array(camMatrix), np.array(distortion),Knew= np.array(camMatrix))
    #undistorted_img = cv2.undistort(output, np.array(camMatrix),distortion, None, newcameramatrix)
                        #from camera matrix undistort image
    #undistorted_img = cv2.undistort(output, np.array(camMatrix), xy_undistorted, None, newcameramatrix)

    return undistorted_img

def resolveHomography(img:np.ndarray, blackLevel:int, whiteLevel:int,hough:int):

    img_vert =  img.copy()
    img_vert = cv2.resize(img_vert, (0, 0), fx = 1, fy = 1)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    img_processed = cv2.GaussianBlur(img.copy(), (33,33),sigmaX=9,sigmaY=9)
    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    #mask region of interest
    maskedPlane = maskingfunction(img_processed,blackLevel, whiteLevel)
    highpass_filter = cv2.cvtColor(high_pass_filter(maskedPlane,33), cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(highpass_filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,10)
    edge_detection = cv2.Canny(binary, 100,150, apertureSize=3)

    #cv2.imshow("Edge", edge_detection)
    #cv2.waitKey(0)

    polar_lines = cv2.HoughLines(edge_detection,1,np.pi/90,hough)

    if polar_lines is not None:
        
        hougLines = drawHoughLines(img,polar_lines)
        #cv2.imshow("Hough Lines", hougLines)
        #cv2.waitKey(0)
        filtered_polar_lines = average_close_pairs(polar_lines,95)

        if len(filtered_polar_lines)==4:
            intersection_pts = lq.hough_lines_intersection(filtered_polar_lines,gray.shape)
            
            intersection_pts = cyclic_intersection_pts(intersection_pts)
            img_corners = img.copy()

            try:
                [drawCircle(img_corners,intersect,12,(0,0,255),6) for intersect in intersection_pts]
                #Draw lines
                edge_detectionshow= cv2.cvtColor(edge_detection,cv2.COLOR_GRAY2BGR)

                grid = np.vstack((np.hstack((maskedPlane,edge_detectionshow)),np.hstack((img_corners,hougLines))))
                img_vert = matchHeight(img_vert, grid)
                grid = np.hstack((grid,img_vert))
                #cv2.imshow('Corners, Mask, Edge, Hough',grid)
                #cv2.imwrite('imageprocessing.png',grid) 
                #cv2.waitKey(0)
            
                #search contours
                contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #cv2.imshow('Find contours',binary)
                #draw bounding box
                rotatedRect = cv2.minAreaRect(contours[0])
                #get rotated rect dimensions
                (x,y), (width, height),angle = rotatedRect
                rotatedRectPts = cv2.boxPoints(rotatedRect)
                rotatedRectPts = np.intp(rotatedRectPts)
                dstPts = [[0,0], [width,0],[width,height], [0,height]]

                #img_corners= cv2.drawContours(img_corners,[rotatedRectPts],0,(0,255,0),2)
                #cv2.imshow('Contours',img_corners)
                #cv2.waitKey(0)
                #transform
                m = cv2.getPerspectiveTransform(np.float32(intersection_pts),np.float32(dstPts))
                #m = cv2.getAffineTransform(np.float32(intersection_pts),np.float32(dstPts))

                #Transform image
                unwarped_img = img.copy()
                #img_copy_dst = cameraUndistordedFunction(img_copy_dst,calibrationPath)
                unwarped_img = cv2.warpPerspective(unwarped_img,m,(int(width), int(height)))
                #unwarped_img = cv2.warpAffine(unwarped_img,m,(int(width), int(height)))
                #draw countour rectangle
                for pts in intersection_pts:
                    cv2.rectangle(unwarped_img, (pts[0] - 1, pts[1] - 1), (pts[0] + 1, pts[1] + 1), (0, 0, 255), 2)
            
                #final_scan = cv2.rotate(unwarped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                final_scan = unwarped_img
                #cv2.imshow('Unwarped',final_scan)
                #print(final_scan.shape)
                #cv2.waitKey(0)
            except:
                final_scan=hougLines
                pass
        else:
            final_scan=hougLines

    return final_scan
        
    #cv2.imwrite('intersect_points.png', img)
       #addcornerpoints(lineOutput[1], undistorted_color_image_output, (255,0,0),(220,34,0))



if __name__ == "__main__":
    main()

   # front = "FrontCover.jpg"
   # #front = "front_cover.jpg"
   # #rear = "RearCover Still.jpg"
   # #calib_path = "Apple_iPhone 12 Pro Max_1x__4k_16by9_3840x2160-30.00fps.json"
   # calib_path="Apple_iPhone 12 Pro Max_0.5x_HD - 30_1080p_16by9_1920x1080-29.99fps.json"
   # ##back = "/Users/adegallaix/Downloads/ocr-video/back_cover.jpg"
   # img = cv2.imread(front)
#
   # img_undistorted = resizeUndistort(img, calib_path)
   # 
   # #img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
   # img_scan = resolveHomography(img, 180 , 200,180)
   # masked = maskingfunction(img_scan, 200,250)
   # highpass_filter = cv2.cvtColor(high_pass_filter(masked,0), cv2.COLOR_BGR2GRAY)
   # binary = cv2.adaptiveThreshold(highpass_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,10)
   # 
   # 
   # d = pytesseract.image_to_data(binary,config='--oem 3 --psm 1 -l fra ',output_type=pytesseract.Output.DICT) 
   # n_boxes = len(d['text'])
   # for i in range(n_boxes):
   #     if int(d['conf'][i]) > 0:
   #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
   #         #cv2.rectangle(resized, (x, y), (x + w, y + h), (75,21,121), -1)
   #         #cv2.putText(resized, f'{d["text"][i]}',(x,y+25),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),1, cv2.LINE_AA)
   #         try:
   #             ps.putBText(img_scan,f'{d["text"][i]}',text_offset_x=x,text_offset_y=y,vspace=2,hspace=1, font_scale=4,background_RGB=(16, 58, 124),text_RGB=(255,255,255),font=1,thickness=1,alpha=0.15,gamma=0.5)
   #         except Exception:
   #             pass
   # binary_view = cv2.cvtColor(binary,cv2.COLOR_BGR2RGB)
   # 
   # photos  = [img_undistorted, masked,binary_view,img_scan]
   # resized_photo = []
   # for photo in photos:
   #     resized_photo.append(matchHeight(photo,img))
   # 
   # hstack_1 = np.hstack((resized_photo[0],resized_photo[1]))
   # hstack_2 = np.hstack((resized_photo[2],resized_photo[3]))
   # print(hstack_1.shape,hstack_2.shape)
   # hstack_2 = cv2.resize(hstack_2, (hstack_1.shape[1], hstack_1.shape[0]),interpolation = cv2.INTER_LINEAR)
   # print(hstack_2.shape)
   # vstack_ = np.vstack((hstack_1,hstack_2))
#
#
   # cv2.imshow('Scan', vstack_)
   # cv2.waitKey(0)
#
   ##cv2.imshow('OCR', img_scan)
   ##cv2.waitKey(0)
   # cv2.destroyAllWindows()
  

    #input_pairs = [[[1.5600000e+02, 1.7104226e+00]], 
    #            [[1.5800000e+02, 1.7104226e+00]],
    #            [[5.9100000e+02, 1.7802358e+00]],
    #            [[1.6790000e+03, 1.0471976e-01]],
    #            [[8.4000000e+02, 3.8397244e-01]],
    #            [[1.6600000e+03, 6.9813170e-02]],
    #            [[8.4300000e+02, 3.8397244e-01]],
    #            [[8.3300000e+02, 3.4906584e-01]],
    #            [[5.4800000e+02, 1.8151424e+00]]
    #           ]
    #print(len(input_pairs))
    #average_close_pairs(input_pairs,95)

    #
    ##img = cv2.imread(back)
###
    #masked = maskingfunction(img, 180,185)
    #highpass_filter = cv2.cvtColor(high_pass_filter(masked,33), cv2.COLOR_BGR2GRAY)
    #binary = cv2.adaptiveThreshold(highpass_filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,10)
    #
    #edge_detection = cv2.Canny(binary, 254, 255, apertureSize=3)
#
    #polar_lines = cv2.HoughLines(edge_detection,1,np.pi/180,285)
#
    #hough_lines = img.copy()
#
    #houghLines=drawHoughLines(hough_lines,polar_lines,"hough_lines.jpg")
#
    #binary = cv2.cvtColor(edge_detection, cv2.COLOR_GRAY2BGR)
    #final_output = np.hstack((masked, img, binary,houghLines))
#
    #cv2.imshow('Masking', final_output)
    #cv2.waitKey(0)
#
    #
    #intersect_pts = lq.hough_lines_intersection(polar_lines, cv2.cvtColor(houghLines,cv2.COLOR_BGR2GRAY).shape)
#
    #intersect_pts = cyclic_intersection_pts(intersect_pts)
#
    #corners = img.copy()
    #[drawCircle(img.copy(),intersect,6,(0,0,255),3) for intersect in intersect_pts]
    #cv2.imshow('Corners',corners)
    #cv2.waitKey(0)
#
    #edge_detection = cv2.cvtColor(edge_detection, cv2.COLOR_GRAY2BGR)
#
    #final_output = np.hstack((masked, img, edge_detection,hough_lines))
    #cv2.imshow('Masking', final_output)
    #cv2.imwrite("Scan.png", final_output)
#
    ###
    #cv2.waitKey(0)
   
