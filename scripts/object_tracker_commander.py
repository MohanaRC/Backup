'''object_displacement_parallel_processing.py : Reads images from a folder/video/url
   Author : Mohana Roy Chowdhury
   First edit : 13th August, 2017'''

import image_loader_main_processor as imp
import cv2
import time
import tagger_custom as tagger
import tagger_using_dumped_images as tagger_frames
import ConfigParser
import object_displacement_parallel_processing as ojd
import os

def read_ini_configuration_files(path_of_config):
    Config = ConfigParser.ConfigParser()
    Config.read(path_of_config)
    roi_sections=Config.sections()
    coordinate_list=[]
    for i, label in enumerate(roi_sections):
        x1=int(Config.get(label, 'upper_left_x'))
        y1=int(Config.get(label, 'upper_left_y')) 
        x2=int(Config.get(label, 'lower_right_x'))
        y2=int(Config.get(label, 'lower_right_y'))
        coordinate_list.append([(x1,y1), (x2, y2)])
    # print coordinate_list
    return coordinate_list

def capture_frames_for_tracking(folder_path, coordinate_list, saving_folder):
    f=[f for f in os.listdir(folder_path)]
    f.sort()
    flag=0
    param_list=[]
    for frame_number, image_path in enumerate(f):
        frame=cv2.imread(folder_path+"/"+image_path)
        if frame_number==0:
            first_image=frame
            # print folder_path+"/"+image_path
            for i, coordinate in enumerate(coordinate_list):
                height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, motion_flag, displacement_counter, additional_text=ojd.get_initial_requirements(first_image, coordinate)
                param_list.append([height, width,dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, motion_flag, displacement_counter, additional_text])                
        imp.main_processor_per_frame(frame, param_list, saving_folder, coordinate_list)



def capture_video_for_tracking(video_path, coordinate_list, saving_folder):
    video=cv2.VideoCapture(video_path)
    flag=0
    param_list=[]
    if video.isOpened():
        rval, frame=video.read()
        first_image=frame
        # print coordinate_list
        for i, coordinate in enumerate(coordinate_list):
            # print coordinate
            height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, motion_flag, displacement_counter, additional_text=ojd.get_initial_requirements(first_image, coordinate)
            param_list.append([height, width,dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, motion_flag, displacement_counter, additional_text])
    else:
        rval=False
    # print param_list
    while rval:
        rval, frame = video.read()
        imp.main_processor_per_frame(frame, param_list, saving_folder, coordinate_list)



def tagging_function_caller(path):
    path_of_config=tagger.capture_frames(path)
    return path_of_config

def control_function():
    print "Please select correct option\n"
    print "For video type 'video'"
    print "For url type 'url'"
    print "For folder path type 'folder'"
    print "For camera enter 'camera'"
    method_of_input = raw_input('Enter desired selection: ')
    print method_of_input
    saving_folder=raw_input('Enter where you want to save the processed images')
    if method_of_input=="video":
        path=raw_input('Enter the path to the video :')
        path_of_config=tagger.capture_frames(path)
        if path_of_config!=None:
            coordinate_list=read_ini_configuration_files(path_of_config)
            capture_video_for_tracking(0, coordinate_list, saving_folder)

    elif method_of_input=="url":
        path=raw_input('Enter the url :')
        path_of_config=tagger.capture_frames(path)
        if path_of_config!=None:
            coordinate_list=read_ini_configuration_files(path_of_config)
            capture_video_for_tracking(0, coordinate_list, saving_folder)        

    elif method_of_input=="folder":
        path=raw_input('Enter the path of the folder where frames are saved :')
        path_of_config=tagger_frames.capture_frames(path)
        if path_of_config!=None:
            coordinate_list=read_ini_configuration_files(path_of_config)
            capture_frames_for_tracking(path, coordinate_list, saving_folder)

    elif method_of_input=="camera":
        path_of_config=tagger.capture_frames(0)
        if path_of_config!=None:
            coordinate_list=read_ini_configuration_files(path_of_config)
            capture_video_for_tracking(0, coordinate_list, saving_folder)


    # path_of_config=tagging_function_caller(path)
    # print path_of_config
    # coordinate_list=read_ini_configuration_files(path_of_config)
    # capture_video_for_tracking(path, coordinate_list)
    



if __name__ =="__main__":
    control_function()

