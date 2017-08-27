'''image_loader_main_processor.py : Processes images and performs tracking of objects
   Author : Mohana Roy Chowdhury
   Date : 13th August, 2017'''



import object_displacement_parallel_processing as odj
import time
import datetime
import threading
import cv2



global text_file
global displacement_counter
global motion_flag
global frame_number
global roi_around_template_list
global height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template,radius, additional_text
global threads

displacement_counter=[]
motion_flag=[]
text_file=open("logger_object_pickup_"+str(datetime.datetime.now())+".txt", "w")
frame_number=0
roi_around_template_list=[]
original_x0=[]
original_y0=[]
original_x1=[]
original_y1=[]
template=[]
radius=[]
additional_text=[]
threads=[]
text=[]
time_displaced=[]
saving_number=0


'''Function name : unpack_param_list 
   Description : Converts list of parameters to individual variables
   Input parameters :
   param_list : List of parameters
   Output parameters :
   height, width, dimension : Height width dimension (3 for RGB, 1 for grayscale) of the image
   first_background : First background for background subtraction 
   original_x0, original_y0, original_x1, original_y1 : x1, y1, x2, y2 coordinates of the template
   template : Cropped template 
   radius : radius shows the area where the object detection would indicate initial position of object
   roi_around_template : ROI coordinates around the object template
   additional_text : additional text to display alarms and other messages'''
def unpack_param_list(param_list):
    height=param_list[0]
    width=param_list[1]
    dimension=param_list[2]
    first_background=param_list[3]
    original_x0=param_list[4]
    original_y0=param_list[5]
    original_x1=param_list[6]
    original_y1=param_list[7]
    template=param_list[8]
    radius=param_list[9]
    roi_around_template=param_list[10]
    # motion_flag=param_list[11]
    # displacement_counter=param_list[12]
    additional_text=param_list[13]
    return height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, additional_text

'''Function name : process_each_thread 
   Description : Processes each frame as multiple threads
   Input parameters : 
   frame : Full image where analysis needs to be done 
   index_of_thread : Index indicating the object on which the processing is being done'''

def process_each_thread(frame, index_of_thread):
    global text_file
    global displacement_counter
    global motion_flag
    global frame_number
    global roi_around_template_list
    global height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template,radius, additional_text
    global threads
    global text
    global time_displaced 
    global saving_number
    text_file.write("Processing frame number and thread number"+str(frame_number)+" "+str(index_of_thread)+"\n")
    saving_number=odj.create_frame_number_for_saving_frame(frame_number)
    # print "Index of thread", index_of_thread
    
    time_displaced[index_of_thread]=""
    text[index_of_thread]=""
    # time1=time.time()
    x_length, y_length, leftx, lefty=odj.template_measures(roi_around_template_list[index_of_thread])
    first_quadrant, second_quadrant, third_quadrant, fourth_quadrant, centre_x, centre_y=odj.convert_roi_crop_into_4_quadrants(roi_around_template_list[index_of_thread])
    # print "quardrant are", first_quadrant, second_quadrant, third_quadrant, fourth_quadrant
    first_quadrant_bgd, second_quadrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg=odj.divide_to_quadrants(first_background, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant)
      
    gray_foreground=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore=odj.divide_to_quadrants(gray_foreground, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant)
    # print first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore
    first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh=odj.perform_background_subtraction_quadrants(first_quadrant_bgd, second_quadrant_bgd, third_quadrant_bgd, fourth_quadrant_bdg, first_quadrant_fore, second_quadrant_fore, third_quadrant_fore, fourth_quadrant_fore)
    percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth=odj.calculate_percentage_motion_all_quadrants(first_quad_thresh, second_quad_thresh, third_quad_thresh, fourth_quad_thresh)
    flag_1, flag_2, flag_3, flag_4=odj.check_for_motion(percentage_motion_first, percentage_motion_second, percentage_motion_third, percentage_motion_fourth)
        ### Do histogram back propagation wherever there is motion#####        
    list_centroids=[0, 0, 0, 0]
    list_numpixels=[0, 0, 0, 0]
    # print "***************", frame_number
    # print flag_1, flag_2, flag_3, flag_4
    if flag_1!=0:
        list_centroids, list_numpixels=odj.flag_processor(frame, first_quadrant, template[index_of_thread], list_centroids, list_numpixels, 0, radius[index_of_thread])
    if flag_2!=0:
        list_centroids, list_numpixels=odj.flag_processor(frame, second_quadrant, template[index_of_thread], list_centroids, list_numpixels, 1, radius[index_of_thread]) 
    if flag_3!=0:
        list_centroids, list_numpixels=odj.flag_processor(frame, third_quadrant, template[index_of_thread], list_centroids, list_numpixels, 2, radius[index_of_thread])
    if flag_4!=0:
        list_centroids, list_numpixels=odj.flag_processor(frame, fourth_quadrant, template[index_of_thread], list_centroids, list_numpixels, 3, radius[index_of_thread])

    # print list_centroids, list_numpixels
    k=[i for i, e in enumerate(list_centroids) if e!=0]
    if len(k)==0 and motion_flag[index_of_thread]==0:
        # print "Object has not been displaced"
        text_file.write("Object"+str(index_of_thread)+"in initial position \n")
        text[index_of_thread]="Object is still in initial position"
        additional_text[index_of_thread]=""
        displacement_counter[index_of_thread]=0
        # to_draw=odj.draw_ROI_rectangles(frame, roi_around_template_list[index_of_thread], (255, 255, 0))
    elif len(k)>0 and motion_flag[index_of_thread]==0:
        displacement_counter[index_of_thread]=0
        additional_text[index_of_thread]=""
        index=max(enumerate(list_numpixels),key=lambda x: x[1])[0]
        time_displaced[index_of_thread]=str(datetime.datetime.now())
        # print index, leftx, lefty, centre_x, centre_y, text, list_centroids
        reconverted_x, reconverted_y, text[index_of_thread]=odj.final_centroid_decision(index, leftx, lefty, centre_x, centre_y, text, list_centroids)
        text_file.write("Index is "+str(index_of_thread)+"\n")
        text_file.write(text[index_of_thread]+"\n")
        text_file.write("Object"+str(index_of_thread)+" displaced at "+time_displaced[index_of_thread]+"\n")
        roi_around_template_list[index_of_thread]=odj.update_ROI(reconverted_x, reconverted_y,x_length, y_length, width, height,leftx, lefty)
        motion_flag[index_of_thread]=1 ## Meaning object has been moved
        # to_draw=draw_ROI_rectangles(frame, roi_around_template_list[index_of_thread], (255, 255, 0))
    elif len(k)>0 and motion_flag[index_of_thread]==1:
        displacement_counter[index_of_thread]=0
        index=max(enumerate(list_numpixels),key=lambda x: x[1])[0]
        # print index, leftx, lefty, centre_x, centre_y, text, list_centroids
        reconverted_x, reconverted_y, text[index_of_thread]=odj.final_centroid_decision(index, leftx, lefty, centre_x, centre_y, text, list_centroids)
        if reconverted_x>=original_x0[index_of_thread] and reconverted_x<=original_x1[index_of_thread] and reconverted_y>=original_y0[index_of_thread] and reconverted_y<=original_y1[index_of_thread]:
            text[index_of_thread]="Object"+str(index_of_thread)+" has been returned"
            motion_flag[index_of_thread]=0
            time_displaced[index_of_thread]=str(datetime.datetime.now())
            text_file.write("Object"+str(index_of_thread)+" returned at "+time_displaced[index_of_thread]+"\n")
            # print "Quad 4", list_centroids[index][1], list_centroids[index][1]+centre_y, list_centroids[index][1]+centre_y+lefty
            # print "leftx, lefty", leftx, lefty
        else :
            text_file.write(text[index_of_thread]+"\n")
        roi_around_template_list[index_of_thread]=odj.update_ROI(reconverted_x, reconverted_y,x_length, y_length, width, height,leftx, lefty)
        # to_draw=odj.draw_ROI_rectangles(frame, roi_around_template_list[index_of_thread], (255, 255, 0))
    elif len(k)==0 and motion_flag[index_of_thread]==1:
        text[index_of_thread]="Object"+str(index_of_thread)+"has been displaced"
        text_file.write(text[index_of_thread]+"\n")
        displacement_counter[index_of_thread]=displacement_counter[index_of_thread]+1
        if displacement_counter[index_of_thread]>=40:
            # to_draw=odj.draw_ROI_rectangles(frame, roi_around_template_list[index_of_thread], (0, 0, 255))
            time_displaced[index_of_thread]=str(datetime.datetime.now())
            additional_text[index_of_thread]="ALARM for object " + str(index_of_thread)
            text_file.write("Object "+str(index_of_thread)+" missing for over 40 frames, generating ALARM \n")
            text_file.write("Current time : "+time_displaced[index_of_thread]+"\n")
        # else:
        #     print " "
            # to_draw=odj.draw_ROI_rectangles(frame, roi_around_template_list[index_of_thread], (255, 255, 0))
            # print "Object displaced or lost"
        # print motion_flag, frame_number
        # print "ROI Updated", roi_around_template
        # to_draw=create_UI(to_draw, text,additional_text, time_displaced)
        # to_draw=draw_ROI_rectangles(frame, roi_around_template, (255, 255, 0))

        # cv2.imwrite("consolidated_tracking_"+folder_path+"/"+"image_"+saving_number+str(frame_number)+".png", to_draw)

        # time2=time.time()
        # lag=time2-time1
        # text_file.write("Processing time"+str(lag)+"\n")
        # text_file.write("\n")  


'''Function name : write_frames
   Description : Writes information in the frames and dumps it in a particular folder
   Input parameters : 
   frame : Image on which information in the frames
   saving_folder : Folder where image has to be saved 
   coordinate_list : List of updated coordinates based on the movement of the objects'''
def write_frames(frame, saving_folder, coordinate_list):
    for index_of_thread, coordi in enumerate(roi_around_template_list):
        if displacement_counter[index_of_thread]>=40:
            to_draw=odj.draw_ROI_rectangles2(frame, coordi, (0, 0, 255))
            to_draw=odj.create_UI(to_draw, text[index_of_thread],additional_text[index_of_thread], time_displaced[index_of_thread])
        elif displacement_counter[index_of_thread]<40 and motion_flag[index_of_thread]==1:
            to_draw=odj.draw_ROI_rectangles2(frame, coordi, (0, 140, 255))
            to_draw=odj.create_UI(to_draw, text[index_of_thread],additional_text[index_of_thread], time_displaced[index_of_thread])
        elif motion_flag[index_of_thread]==0 and displacement_counter[index_of_thread]==0:
            to_draw=odj.draw_ROI_rectangles2(frame, coordi, (0, 255, 0))
        cv2.imwrite(saving_folder+"/"+"image_"+saving_number+str(frame_number)+".png", to_draw)



'''Function name : main_processor_per_frame
   Description : Main processor for the frame 
   Input parameters :
   frame : Image on which information has to be processed 
   param_list : List of lists containing all the required parameters of the image 
   saving_folder : Folder where images have to be saved 
   coordinate_list : List having coordinates '''
def main_processor_per_frame(frame, param_list, saving_folder, coordinate_list):
    global text_file
    global displacement_counter
    global motion_flag 
    global frame_number
    global roi_around_template_list
    global height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template,radius, additional_text
    global threads
    global text
    global time_displaced
    threads=[]
    
    frame_number=frame_number+1
    print "Frame number", frame_number
    # height, width, dimension, first_background, original_x0, original_y0, original_x1, original_y1, template, radius, roi_around_template, additional_text=unpack_param_list(param_list)
    text_file.write("Processing frame number "+str(frame_number)+"\n")
    # time_displaced=[]
    # text=[]
    time1=time.time()
        ### I save all the original x0, y0 etc values there 
    if frame_number==1:
        for i, param_list_values in enumerate(param_list):

            height, width, dimension, first_background, x0, y0, x1, y1, temp, rad, roi_around_template, add_text=unpack_param_list(param_list_values)
            original_x0.append(x0)
            original_y0.append(y0)
            original_x1.append(x1)
            original_y1.append(y1)
            template.append(temp)
            radius.append(rad)
            roi_around_template_list.append(roi_around_template)
            motion_flag.append(0)
            text.append("")
            time_displaced.append("")
            displacement_counter.append(0)
            additional_text.append("")
            # additional_text.append(add_text)            

    for i in range(0, len(param_list)):
        threads.append((threading.Thread(target=process_each_thread(frame, i))))
        threads[i].start()
    for i in range(0, len(threads)):    
        threads[i].join()
    time2=time.time()
    time_interval_total=time2-time1
    # print "Time to process this frame", frame_number
    text_file.write("Overall processing time is "+str(time_interval_total)+"\n")
    write_frames(frame, saving_folder, coordinate_list)
    time_interval_write=time.time()-time1
    text_file.write("Including writing processing time is "+str(time_interval_write)+"\n")
    text_file.write("   \n")




# text_file.close()

    





if __name__ =="__main__":
    folder_path = "frameset4"

    # main_processesing_function(folder_path) 
    main_processor(folder_path)   
