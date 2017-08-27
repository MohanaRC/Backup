'''read_images.py : Reads images from a folder containing frames from a video
   Author : Mohana Roy Chowdhury
   First edit : 18th July, 2017'''

import cv2
import os
import handle_mouse_clicks ## this function is used for the tagging. Handles mouse events
import numpy as np 
import supporting_functions_for_tracking ## has the supporting functions for kl tracking
import supporting_function_for_denseoptical_flow

def draw_ROI_rectangles(image, rects, color):
    cv2.rectangle(image, rects[0], rects[1], color, 2)
    return image

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

def crop_any_image(image, rectangle_coords):
    cropped_image=image[rectangle_coords[0][1]:rectangle_coords[1][1], rectangle_coords[0][0]:rectangle_coords[1][0]]
    return cropped_image

def convert_rgb_to_gray(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return frame_gray

def draw_circle_on_list_of_points(image, list_of_points):
    for x, y in [np.int32(tr[-1]) for tr in list_of_points]:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)    
    return image

def read_image_dense_optical_flow(folder_path):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    first_image=cv2.imread(folder_path+"/"+f[0])
    prvs = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_image)
    print "Area 1"
    hsv[...,1] = 255
    print "Area 2"
    # copy_first_image=first_image.copy()
    (height, width, dimension)=first_image.shape
    print "area 3"
    bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    print "area 4"
    f.remove(f[0]) ### removing first fram
    counter =0
    for image_path in f:
        print "area 5", counter
        frame2=cv2.imread(folder_path+"/"+image_path)
        print "area 6"
        bgr=supporting_function_for_denseoptical_flow.optical_flow_movie(prvs,frame2, hsv)
        # cv2.imshow('frame2',bgr)
        image_together=np.hstack((frame2, bgr))
        if cv2.waitKey(0):
            cv2.imshow("image", image_together)
            cv2.imwrite("dense_opflow_"+folder_path+"/"+image_path, image_together)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png',frame2)
        #     cv2.imwrite('opticalhsv.png',bgr)
        prvs=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        counter=counter+1







def read_images_from_folder(folder_path):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    first_image=cv2.imread(folder_path+"/"+f[0])
    copy_first_image=first_image.copy()
    (height, width, dimension)=first_image.shape
    bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    object_initial_crop=crop_any_image(copy_first_image, bin_roi)### Get crops of the object
    # cv2.imwrite("image_crop.png", object_initial_crop) 
    '''Initiating the tracking process'''
    object_intial_gray_crop=convert_rgb_to_gray(object_initial_crop)### Convert the crop to grayscale
    ###Initialize the tracking parameters
    track_len, detect_interval, tracks, frame_idx=supporting_functions_for_tracking.kl_tracking_initial_values()
    lk_params, feature_params=supporting_functions_for_tracking.kl_feature_params_initializer()
    # # get the gray crop where the points to track lie
    tracks, p = supporting_functions_for_tracking.initializer_tracking_kl(object_intial_gray_crop, tracks, lk_params, feature_params, bin_roi)
    # first_image=draw_circle_on_list_of_points(first_image, tracks)
    # print "tracks ", tracks, type(tracks)
    # print "p ", p, type(p)
    cv2.imwrite("image_crop.png", first_image) 
    roi_coordinates=create_roi_from_crop_coordinates(height, width, bin_roi)    
    # print "rects_roi is ", bin_roi
    image_counter=0
    for image_path in f:
        raw_image=cv2.imread(folder_path+"/"+image_path)      
        image=draw_ROI_rectangles(raw_image, roi_coordinates, (255, 255, 0)) #draw displacement ROI
        if image_counter==0:
            image=draw_ROI_rectangles(image, bin_roi, (0, 255, 0))#draw the object ROI
            first_image=draw_circle_on_list_of_points(image, tracks)
        
        if image_counter>0:
            img0=convert_rgb_to_gray(prev_frame)
            img1=convert_rgb_to_gray(raw_image)
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            print "p0",p0
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            print "p1", p1
            print "st", st
            print "err", err
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            print "p0r", p0r
            print "st", st
            print "err", err
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            print "d", d
            good = d < 100
            print "good", good
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(image, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        prev_frame=raw_image
        image_counter=image_counter+1
        if cv2.waitKey(0):
            cv2.imshow("image", image)  
            print "klt_tracking_"+folder_path+"/"+image_path
            cv2.imwrite("klt_tracking_"+folder_path+"/"+image_path, image)

        # key = cv2.waitKey(0) 

def read_image_background_subtraction(folder_path):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    first_image=cv2.imread(folder_path+"/"+f[0])  ##get first frame
    gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY) #convert to gray
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0) # gaussian blurring
    copy_first_image=first_image.copy() # make a copy of the first frame
    image_counter=0
    for image_path in f:
        frame=cv2.imread(folder_path+"/"+image_path)
        gray=cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        # gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)             
        if image_counter==0:
            firstFrame = gray
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        image, cnts, hierachy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
        # loop over the contours
        print "cnt", cnts       
        for c in cnts:
            print "Printing c", c
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
 
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # show the frame and record if the user presses a key
        thresh2=cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        image_together=np.hstack((frame, thresh2))
        frameDelta2=cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR)
        image_together_full=np.hstack((image_together, frameDelta2))
        # cv2.imshow("Security Feed", frame)
        # cv2.imshow("Thresh", thresh)
        # if cv2.waitKey(0):
        #     cv2.imshow("Frame Delta", image_together_full)
        # # key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key is pressed, break from the lop
        # if key == ord("q"):
        #     break

def create_motion_history_mask(mask, motion_history, factor):
    # motion_history=motion_history.point(lambda i: i * factor)
    motion_history=motion_history/factor
    print type(motion_history), type(mask)
    motion_history_updated=cv2.absdiff(motion_history, mask)
    # motion_history=cv2.addWeighted(motion_history, factor, mask, 1, 0)
    # mask=motion_history
    return motion_history_updated


def read_images_for_motion_template(folder_path):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    first_image=cv2.imread(folder_path+"/"+f[0])  ##get first frame
    copy_frame=first_image.copy()
    gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY) #convert it to background
    (height, width, dimension)=first_image.shape
    background_frame=gray
    bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    '''The template below is the template for template matching'''
    object_initial_crop=crop_any_image(copy_frame, bin_roi)### Get crops of the object 
    cv2.imwrite("object_initial.png", object_initial_crop)
    '''This is the area for background subtraction'''
    roi_coordinates=create_roi_from_crop_coordinates(height, width, bin_roi)  
    roi_background=crop_any_image(gray, roi_coordinates)   
    counter=1
    for image_path in f:
        frame=cv2.imread(folder_path+"/"+image_path)
        gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        roi_foreground=crop_any_image(gray_frame, roi_coordinates)
        roi_foreground_color=crop_any_image(frame, roi_coordinates)
        
        if counter==5:
            cv2.imwrite("roi_foreground.png", roi_foreground_color)
            break
        counter=counter+1
        frameDelta = cv2.absdiff(roi_background, roi_foreground)
        thresh = cv2.threshold(frameDelta, 60, 255, cv2.THRESH_BINARY)[1]
        # print np.count_nonzero(thresh)
        percentage_motion= (np.count_nonzero(thresh)*1.0/(thresh.shape[0]*thresh.shape[1]))*100
        if percentage_moti 


            

        # contours= cv2.findContours(thresh, 1, 2)
        # print contours
        # cnt = contours[0]
        # rows,cols = img.shape[:2]

        # cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
        # x,y,w,h = cv2.boundingRect(cnt)
        # x=x+roi_coordinates[0][0]
        # y=y+roi_coordinates[0][1]
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((cols-x)*vy/vx)+y)
        # cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
        # mask=thresh
        # print mask.shape
        # motion_history=create_motion_history_mask(mask, motion_history, factor)
        background_frame=gray_frame
        cv2.imwrite("background_subtraction_"+folder_path+"/"+image_path, thresh)





    # (height, width, dimension)=first_image.shape
    # bin_roi=handle_mouse_clicks.get_ROI_for_image(first_image)
    # object_initial_crop=crop_any_image(copy_first_image, bin_roi)    







if __name__ =="__main__":
    folder_path = "frameset4"
    # read_images_from_folder(folder_path)
    # read_image_dense_optical_flow(folder_path)
    # read_image_background_subtraction(folder_path)
    read_images_for_motion_template(folder_path)