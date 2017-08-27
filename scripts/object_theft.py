'''read_images.py : Reads images from a folder containing frames from a video
   Author : Mohana Roy Chowdhury
   First edit : 25th July, 2017'''

import cv2
import os
import handle_mouse_clicks 
import numpy as np 
from matplotlib import pyplot as plt
from skimage import measure
import math
import time


'''Function name : draw_ROI_rectangles
   Description : Draws roi using opencv cv
   Input parameters :
   image : Image on with drawing has to be done 
   rects : Top left bottom right coordinate of the rectange in form of [(x1,y1), (x2,y2)]
   color : BRG colors as a tuple (B, G, R)
   Output parameters :
   image : Image with rectangle drawn on it'''
def draw_ROI_rectangles(image, rects, color):
    cv2.rectangle(image, rects[0], rects[1], color, 2)
    return image

'''Function name : create_roi_from_crop_coordinates
   Description : Given a template it finds a region of interest around the template taking into consideration the boundaries of the full image
   Input parameters : 
   height : Height of the full image
   width : width of the full image
   bin_roi : Top left bottom right coordinates of the template in the form of [(x1,y1), (x2,y2)]
   Output parameters :
   roi_coordinates : Top left bottom right coordinates of the ROI in form of [(x1,y1), (x2, y2)]'''
def create_roi_from_crop_coordinates(height, width, bin_roi):
    x1=bin_roi[0][0]
    y1=bin_roi[0][1]
    x2=bin_roi[1][0]
    y2=bin_roi[1][1]
    extentx=(x2-x1)*2
    extenty=(y2-y1)*2
    roi_x1=max(0, (x1-extentx))
    roi_x2=min(width, (x2+extentx))
    roi_y1=max(0, (y1-extenty))
    roi_y2=min(height, (y2+extenty))
    roi_coordinates=[(roi_x1, roi_y1), (roi_x2, roi_y2)]
    return roi_coordinates

'''Function name : convert_roi_crop_into_4_quadrants
   Description : Coverts the ROI into 4 quadrants and sends the coordinates 
   Input parameters : 
   bin_roi :(here) Top left bottom right coordinates of the ROI in the form of [(x1,y1), (x2,y2)]
   Output parameters :
   first_quadrant : Top left bottom right coordinates of the first quadrant (same format as mentioned before)
   second_quadrant : Top left bottom right coordinates of the second quardrant (same format as mentioned before)
   third_quadrant : Top left bottom right coordinates of the third quardrant (same format as mentioned before)
   fourth_quadrant : Top left bottom right coordinates of the fourth quardrant (same format as mentioned before)'''
def convert_roi_crop_into_4_quadrants(bin_roi):
    x1=bin_roi[0][0]
    y1=bin_roi[0][1]
    x2=bin_roi[1][0]
    y2=bin_roi[1][1]
    centre_x=(x1+x2)/2
    centre_y=(y1+y2)/2
    first_quadrant=[(x1,y1), (centre_x, centre_y)]
    second_quadrant=[(centre_x, y1), (x2, centre_y)]
    third_quadrant=[(centre_x, centre_y), (x2,y2)]
    fourth_quadrant=[(x1, centre_y), (centre_x, y2)]
    return first_quadrant, second_quadrant, third_quadrant, fourth_quadrant, centre_x, centre_y

'''Function name : crop_any_image
   Description : Takes and image and crops it based on the coordinates send in the same standard format used in our code 
   Input parameters : 
   image : Image from which the crop has to be made 
   rectangle_coords : Coordinates of the crops in the same format as mentioned above 
   Output parameters :
   cropped_image : Cropped image'''
def crop_any_image(image, rectangle_coords):
    cropped_image=image[rectangle_coords[0][1]:rectangle_coords[1][1], rectangle_coords[0][0]:rectangle_coords[1][0]]
    return cropped_image

'''Function name : convert_rgb_to_gray
   Description : Convert rgb images to gray
   Input parameters :
   image : rgb image 
   Output parameters :
   frame_gray : Grayscale images'''
def convert_rgb_to_gray(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return frame_gray

def draw_circle_on_list_of_points(image, list_of_points):
    for x, y in [np.int32(tr[-1]) for tr in list_of_points]:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)    
    return image

def divide_to_quadrants(image, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant):
    first_quad_image=crop_any_image(image, first_quadrant)
    second_quad_image=crop_any_image(image, second_quadrant)
    third_quad_image=crop_any_image(image, third_quadrant)
    fourth_quad_image=crop_any_image(image, fourth_quadrant)
    return first_quad_image, second_quad_image, third_quad_image, fourth_quad_image

def perform_background_subtraction(fore, bgd):
    frameDelta = cv2.absdiff(bgd, fore)
    thresh = cv2.threshold(frameDelta, 60, 255, cv2.THRESH_BINARY)[1]   
    return frameDelta, thresh

def perform_background_subtraction_quadrants(first_quadrant_bgd, second_quadrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg, first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore):
    first_quad_delta, first_quad_thresh=perform_background_subtraction(first_quadrant_fore, first_quadrant_bgd)
    second_quad_delta, second_quad_thresh=perform_background_subtraction(second_quadrant_fore, second_quadrant_bgd)
    third_quad_delta, third_quad_thresh=perform_background_subtraction(third_quadrant_fore, third_quadrant_bgd)
    fourth_quad_delta, fourth_quad_thresh=perform_background_subtraction(fourth_quadrant_fore, fourth_quadrant_bdg)
    return first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh

# def find_movement_in_quad(threshold_image):

def save_image_and_check(first_quadrant, second_quadrant, third_quadrant, fourth_quadrant, category):
    cv2.imwrite("first_quadrant_"+category+".png", first_quadrant)
    cv2.imwrite("second_quadrant_"+category+".png", second_quadrant)
    cv2.imwrite("third_quadrant_"+category+".png", third_quadrant)
    cv2.imwrite("fourth_quadrant_"+category+".png", fourth_quadrant)

def find_percentage_motion(thresholded_image):
    # print "Thresholded image", thresholded_image.shape 
    size_of_image=thresholded_image.shape[0]*thresholded_image.shape[1]
    number_pixels_moved=np.count_nonzero(thresholded_image)
    percentage_motion=(number_pixels_moved*1.0/size_of_image)*100
    # print size_of_image, number_pixels_moved, percentage_motion
    return percentage_motion

def calculate_percentage_motion_all_quadrants(first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh):
    # print "first_quad_thresh", first_quad_thresh.shape
    # print "second_quad_thresh", second_quad_thresh.shape
    # print "third_quad_thresh", third_quad_thresh.shape
    # print "fourth_quad_thresh", fourth_quad_thresh.shape
    percentage_motion_first=find_percentage_motion(first_quad_thresh)
    percentage_motion_second=find_percentage_motion(second_quad_thresh)
    percentage_motion_third=find_percentage_motion(third_quad_thresh)
    percentage_motion_fourth=find_percentage_motion(fourth_quad_thresh)
    # print "Percentage motion are ", percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth
    return percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth

def check_for_motion(percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth):
    flag_1=0
    flag_2=0
    flag_3=0
    flag_4=0
    if percentage_motion_first>15.0:
        flag_1=1
    if percentage_motion_second>15.0:
        flag_2=1
    if percentage_motion_third>15.0:
        flag_3=1
    if percentage_motion_fourth>15.0:
        flag_4=1
    return flag_1, flag_2, flag_3, flag_4

def histogram_backpropagation(roi, target):
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    #target is the image we search in
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    # Find the histograms using calcHist. Can be done with np.histogram2d also
    M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    # print np.count_nonzero(M)
    I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    # print np.count_nonzero(I)
    R=M/I
    h,s,v = cv2.split(hsvt)
    B = R[h.ravel(),s.ravel()]
    B = np.minimum(B,1)
    B = B.reshape(hsvt.shape[:2])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(B,-1,disc,B)
    B = np.uint8(B)
    cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
    ret,thresh = cv2.threshold(B,50,255,0)
    return thresh
    
def create_mask(thresh, pixel_white_threshold):
    labels=measure.label(thresh, neighbors=8, background=0)
    mask=np.zeros(thresh.shape, dtype="uint8")
    pixel_count=0
    for label in np.unique(labels):
        # print "label",label
        if label==0:
            continue
        labelMask=np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels==label]=255
        numPixels=cv2.countNonZero(labelMask)
        # print numPixels
        if numPixels>pixel_white_threshold:
            mask=cv2.add(mask, labelMask)
            pixel_count=numPixels
    return mask, pixel_count

def create_centroid(mask):
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    centroids=None
    if len(contours)!=0:
        moments=[cv2.moments(cnt) for cnt in contours]
        centroids=[(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in moments]
        centroids=centroids[len(centroids)-1]
    return centroids

def post_motion_processing(frame, quadrant, template):
    area=template.shape[0]*template.shape[1]
    pixel_white_threshold=area/10
    # print "threshold", pixel_white_threshold

    quadrant_color=crop_any_image(frame, quadrant)
    locator_image_thresholded=histogram_backpropagation(template, quadrant_color)
    mask, numPixels=create_mask(locator_image_thresholded, pixel_white_threshold)
    centroids=create_centroid(mask)
    return centroids,mask, numPixels

    # cv2.imwrite("locator_1.png", locator_image_first_quad)

def find_distance(x,y):
    distance=math.sqrt(x**2+y**2)
    return distance 

def template_measures(roi_around_template):
    x_length=roi_around_template[1][0]-roi_around_template[0][0]
    y_length=roi_around_template[1][1]-roi_around_template[0][1]
    return x_length, y_length, roi_around_template[0][0], roi_around_template[0][1]

def update_ROI(reconverted_x, reconverted_y,x_length, y_length, width, height, leftx, lefty):
    # print "Updating ROI"
    above_x=x_length/2
    above_y=y_length/2
    x0=max(0,reconverted_x-above_x)
    y0=max(0, reconverted_y-above_y)
    x1=min(width, reconverted_x+above_x)
    y1=min(height,reconverted_y+above_y)
    roi_around_template=[(x0, y0), (x1, y1)]
    return roi_around_template



def main_processesing_function(folder_path):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    first_image=cv2.imread(folder_path+"/"+f[0])
    (height, width, dimension)=first_image.shape
    first_background=convert_rgb_to_gray(first_image)
    bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    template=crop_any_image(first_image, bin_roi)### Get crops of the object
    radius=min(template.shape[0], template.shape[1])/2
    # print "Template", template.shape[0], template.shape[1], template.shape[2], radius
    roi_around_template=create_roi_from_crop_coordinates(height, width, bin_roi)
    print "*******************************", roi_around_template
    # first_quadrant, second_quadrant, third_quadrant, fourth_quadrant=convert_roi_crop_into_4_quadrants(roi_around_template)
    # first_quad=crop_any_image(first_image, first_quadrant)
    motion_flag=0
    #### Four zones where motion has to be checked
    # print first_quadrant_bgd
    # save_image_and_check(first_quadrant_bgd, second_quardrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg, "background")
    '''Till here I have template and coordis, roi and coordis, quadrants and coordis'''
    for frame_number, image_path in enumerate(f):
        time1=time.time()
        x_length, y_length, leftx, lefty=template_measures(roi_around_template)

        first_quadrant, second_quadrant, third_quadrant, fourth_quadrant, centre_x, centre_y=convert_roi_crop_into_4_quadrants(roi_around_template)
        # print "quardrant are", first_quadrant, second_quadrant, third_quadrant, fourth_quadrant
        first_quadrant_bgd, second_quadrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg=divide_to_quadrants(first_background, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant)

        print "Frame number", frame_number
        frame=cv2.imread(folder_path+"/"+image_path)
      
        gray_foreground=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore=divide_to_quadrants(gray_foreground, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant)
        # print first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore
        first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh=perform_background_subtraction_quadrants(first_quadrant_bgd, second_quadrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg, first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore)
        percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth=calculate_percentage_motion_all_quadrants(first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh)
        flag_1, flag_2, flag_3, flag_4=check_for_motion(percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth)
        ### Do histogram back propagation wherever there is motion#####        
        list_centroids=[0, 0, 0, 0]
        list_numpixels=[0, 0, 0, 0]
        if flag_1!=0:
            # print "First quardrant for tracking"
            centroids, mask, numPixels=post_motion_processing(frame, first_quadrant, template)
            # print centroids, mask
            if centroids!=None:
                distance=find_distance(centroids[0], centroids[1])
                if distance>radius:
                    list_centroids[0]=centroids
                    list_numpixels[0]=numPixels


            # cv2.imwrite("mask_test1.png", mask)
        if flag_2!=0:
            # print "Second quadrant for tracking"
            centroids, mask, numPixels=post_motion_processing(frame, second_quadrant, template)
            # print centroids
            if centroids!=None:
                distance=find_distance(centroids[0], centroids[1])
                if distance>radius:
                    list_centroids[1]=centroids
                    list_numpixels[1]=numPixels

            # cv2.imwrite("mask_test2.png", mask)
        if flag_3!=0:
            # print "Third quadrant for tracking"
            centroids, mask, numPixels=post_motion_processing(frame, third_quadrant, template)
            # print centroids
            if centroids!=None:
                distance=find_distance(centroids[0], centroids[1])
                if distance>radius:
                    list_centroids[2]=centroids
                    list_numpixels[2]=numPixels

            cv2.imwrite("mask_test3.png", mask)
        if flag_4!=0:
            # print "Fourth quadrant for tracking"
            centroids, mask, numPixels=post_motion_processing(frame, fourth_quadrant, template)
            # print centroids
            if centroids!=None:
                distance=find_distance(centroids[0], centroids[1])
                if distance>radius:
                    list_centroids[3]=centroids
                    list_numpixels[3]=numPixels


            # cv2.imwrite("mask_test4.png", mask)
        print list_centroids, list_numpixels
        k=[i for i, e in enumerate(list_centroids) if e!=0]
        if len(k)==0 and motion_flag==0:
            print "Object has not been displaced"
        elif len(k)>0 and motion_flag==0:
            index=max(enumerate(list_numpixels),key=lambda x: x[1])[0]
            print "Motion observed in quadrant", str(index+1)
            if index==0:
                reconverted_x=list_centroids[index][0]+leftx
                reconverted_y=list_centroids[index][1]+lefty
            elif index==1:
                reconverted_x=list_centroids[index][0]+centre_x
                reconverted_y=list_centroids[index][1]+lefty
            elif index==2:
                reconverted_x=list_centroids[index][0]+centre_x
                reconverted_y=list_centroids[index][1]+centre_y
                print "Quad 3", list_centroids[index][1], list_centroids[index][1]+centre_y, list_centroids[index][1]+centre_y+lefty
            elif index==3:
                reconverted_x=list_centroids[index][0]+leftx
                reconverted_y=list_centroids[index][1]+centre_y
                print "Quad 4", list_centroids[index][1], list_centroids[index][1]+centre_y, list_centroids[index][1]+centre_y+lefty
            print "left x, left y", leftx, lefty

            roi_around_template=update_ROI(reconverted_x, reconverted_y,x_length, y_length, width, height,leftx, lefty)
            motion_flag=1 ## Meaning object has been moved
        elif len(k)>0 and motion_flag==1:
            index=max(enumerate(list_numpixels),key=lambda x: x[1])[0]

            print "Object being tracked to quadrant", str(index+1)
            if index==0:
                reconverted_x=list_centroids[index][0]+leftx
                reconverted_y=list_centroids[index][1]+lefty
            elif index==1:
                reconverted_x=list_centroids[index][0]+centre_x
                reconverted_y=list_centroids[index][1]+lefty
            elif index==2:
                reconverted_x=list_centroids[index][0]+centre_x
                reconverted_y=list_centroids[index][1]+centre_y
                print "Quad 3", list_centroids[index][1], list_centroids[index][1]+centre_y, list_centroids[index][1]+centre_y+lefty
            elif index==3:
                reconverted_x=list_centroids[index][0]+leftx
                reconverted_y=list_centroids[index][1]+centre_y
                print "Quad 4", list_centroids[index][1], list_centroids[index][1]+centre_y, list_centroids[index][1]+centre_y+lefty
            print "leftx, lefty", leftx, lefty
            roi_around_template=update_ROI(reconverted_x, reconverted_y,x_length, y_length, width, height,leftx, lefty)
        
        elif len(k)==0 and motion_flag==1:
            print "Object displaced or lost"

        # print "ROI Updated", roi_around_template
        to_draw=draw_ROI_rectangles(frame, roi_around_template, (255, 255, 0))
        # if frame_number<10:
        #     # cv2.imwrite("consolidated_tracking_"+folder_path+"/"+"image_00"+str(frame_number)+".png", to_draw)
        # elif frame_number>=10 and frame_number<100:
        #     # cv2.imwrite("consolidated_tracking_"+folder_path+"/"+"image_0"+str(frame_number)+".png", to_draw)
        # else:
        #     # cv2.imwrite("consolidated_tracking_"+folder_path+"/"+"image_"+str(frame_number)+".png", to_draw)

        time2=time.time()
        print "time required", time2-time1
        # print str(frame_number)+" "+str(flag_1)+" "+str(flag_2)+" "+str(flag_3)+" "+str(flag_4)
        # save_image_and_check(first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh, "threshold")

        # if frame_number==15:
        #     break




if __name__ =="__main__":
    folder_path = "frameset4"

    main_processesing_function(folder_path)    



    # '''Initiating the tracking process'''
    # object_intial_gray_crop=convert_rgb_to_gray(object_initial_crop)### Convert the crop to grayscale
    # ###Initialize the tracking parameters
    # track_len, detect_interval, tracks, frame_idx=supporting_functions_for_tracking.kl_tracking_initial_values()
    # lk_params, feature_params=supporting_functions_for_tracking.kl_feature_params_initializer()
    # # # get the gray crop where the points to track lie
    # tracks, p = supporting_functions_for_tracking.initializer_tracking_kl(object_intial_gray_crop, tracks, lk_params, feature_params, bin_roi)
    # # first_image=draw_circle_on_list_of_points(first_image, tracks)
    # # print "tracks ", tracks, type(tracks)
    # # print "p ", p, type(p)




    # prvs = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(first_image)
    # print "Area 1"
    # hsv[...,1] = 255
    # print "Area 2"
    # # copy_first_image=first_image.copy()
    # (height, width, dimension)=first_image.shape
    # print "area 3"
    # bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    # print "area 4"
    # f.remove(f[0]) ### removing first fram
    # counter =0
    # for image_path in f: