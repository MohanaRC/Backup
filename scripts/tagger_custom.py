import cv2
import configparser
from copy import deepcopy

'''Function name : give_preview 
   Description : Shows a sample image to explain the process of tagging'''
def give_preview():
    print "This is the tagging script. Please go through the instructions and sample image before proceeding with the tagging process"
    print "The image below shows a sample tagging image"
    print "Press zero once you have understood how to tag"
    print "Please note : The tagging process works better for larger objects so please tag accordingly"
    sample=cv2.imread("image_crop.png")
    print sample.shape
    cv2.imshow("image",sample)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
    return 0

'''Function name : print_instructions_for_selecting_frame
   Description : Prints out a set of instructions for frame selection'''
def print_instructions_for_selecting_frame():
    print "The set of instructions are as follows:"
    print "The code will run you through a set of frames from a video. Check if a frame is suitable for tagging"
    print "If you like a particular frame for the tagging process press \n c to CONFIRM the frame and \n s to SKIP the frame"
    print "Once you press s you the next frame will be loaded"
    print "Once you press c the frame will be locked for tagging"

'''Function name : capture_video
   Description : Captures a video from camera or path and runs through the frames to select a particular frame for tagging
   Input Parameters :
   video_path : Path of the video on which tagging is to be done
   Output Parameters :
   locked_frame : Frame confirmed for the tagging process
   flag : Flag value confirms if a frame has been confirmed or all frames have been skipped'''
def capture_video(video_path):
    video=cv2.VideoCapture(video_path)
    locked_frame=None
    flag=0
    if video.isOpened():
        rval, frame=video.read()
    else:
        rval=False
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = video.read()
        key = cv2.waitKey(0)
        if key == ord("s"):
            print "Skipping current frame"
            continue
        elif key == ord("c"):
            print "Selection has been made"
            locked_frame=frame
            cv2.destroyAllWindows()
            flag=1
            break
        elif key == ord("q"):
            print "Cancelling process"
            cv2.destroyAllWindows()
            quit()
    if flag==0:
        print "No frame selected in the video"
        return None, flag
    else:
        return locked_frame, flag
            
cv2.destroyWindow("preview")    

'''Function name : print_instructions_for_selecting_ROI
   Description : Prints a set of instructions for selecting ROI'''
def print_instructions_for_selecting_ROI():
    print "Please follow these instructions to select the ROI"
    print "Draw a rectangle by pressing left mouse button on the top left and dragging it till bottom right and releasing it"
    print "Press c if you want to confirm, press r if you want to reset. If you press c, that particular box will turn red"
    print "If you want to select another box press y else print n"
    print "Once you press n the coordinates of all the rectangles will get dumped as a config file"
    print "The path to the config file will be displayed"


 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

'''Function name : click_and_crop
   Description : Identifies mouse clicking and identifies the pixels being clicked on
   Input Parameters :
   event, x, y, flags, param:Internal values being computed for handling mouse events, x, y are coordinates
   Output Parameters :
   None since refPt is global'''

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        

'''Function name : get_ROI_for_image
   Description : Gets the roi for the image
   Input Parameters : 
   image : Image where tagging is being done
   coordinates : List holding coordinates
   Output Parameters
   coordinates : Updated list of coordinates
   image : Updated image with confirmed rectange'''

def get_ROI_for_image(image, coordinates):
    # print type(coordinates)
    global refPt
    refPt=[]
    # print "Back here"
    clone=image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    counter=1
    while True:
    # print "Inside while loop and displaying the image"
    # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF 
        ### When two coordinates are obtained we draw a rectange
        if len(refPt)==2:
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)    
        key = cv2.waitKey(1) & 0xFF    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            print "Pressing r, resetting the cropping region"
            image = clone.copy()
            
            # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            print "Pressing c and breaking off from the loop"
            image=cv2.rectangle(image, refPt[0], refPt[1], (0, 0, 255), 2)
            cv2.destroyAllWindows()
            # print "refPt is ", refPt, type(refPt)
            # print "coordinates", type(coordinates)
            coordinates.append(refPt)
            # print "Appended", coordinates
            return coordinates, image
            # cv2.imshow("image", image) 
            break

'''Function name : save_config_file
   Description : Saves the coordinates in a config file 
   Input Parameters :
   config : config file 
   x1 : Top left x coordinate
   y1 : Top left y coordinate 
   x2 : Bottom right x coordinate 
   y2 : Bottom right y coordinate
   Output Parameters
   config : Updated config file '''  

def save_config_file(config, x1, y1, x2, y2, count):
    config['ROI'+str(count)]['upper_left_x'] = str(x1)
    config['ROI'+str(count)]['upper_left_y'] = str(y1)
    config['ROI'+str(count)]['upper_right_x'] = str(x2)
    config['ROI'+str(count)]['upper_right_y'] = str(y1)
    config['ROI'+str(count)]['lower_right_x'] = str(x2)
    config['ROI'+str(count)]['lower_right_y'] = str(y2)
    config['ROI'+str(count)]['lower_left_x'] = str(x1)
    config['ROI'+str(count)]['lower_left_y'] = str(y2)
    return config

'''Function name : construct_full_rectangles
   Description : Get all four coordinates of rectangles given two points and creates and dumps a config file for the same
   Input Parameters :
   rectangle_coordis : List containing top left and bottom right as [[(x1, y1), (x2, y2)],[(x3,y3), (x4,y4)]]
   Output Parameters :
   full_coordis : List having full coordinates saved in the same manner as shown above'''
def construct_full_rectangles(rectangle_coordis):
    config = configparser.ConfigParser()
    config.optionxform = str
    path_to_save = 'config.ini'

    full_coordis=[]
    for count, i in enumerate(rectangle_coordis):
        # config['ROI'] = {}

        config['ROI'+str(count+1)]={}
        x1=i[0][0]
        y1=i[0][1]
        x2=i[1][0]
        y2=i[1][1]
        number=count+1
        config=save_config_file(config, x1, y1, x2, y2, number)
        full_coordis.append([(x1,y1), (x2, y1), (x2, y2), (x1,y2)])
    with open(path_to_save, 'w') as configfile:
        config.write(configfile)
    print "Configuration saved at "+path_to_save
    return full_coordis, path_to_save

'''Function name : capture_frames
   Description : Coodinates the entire tagging process
   Steps : Displays all the necessary instructions and sample image frame
           Makes the user select a particular frame where tagging has to be done
           Opens a window displaying the image and records the tags
           Converts the top left and bottom right coordinates as full rectangle coordinates
           Dumps the configuration as a config file'''
def capture_frames(video_path):
    path_to_save=None
    give_preview()
    print_instructions_for_selecting_frame()
    frame, flag=capture_video(video_path)
    if flag == 0 :
        print "Cannot proceed with ROI selection as frame has not been selected"
        quit()
    else:
        print_instructions_for_selecting_ROI()
        further_ROIs=0
        coordinates=[]

        while further_ROIs==0:
            print "**********"
            coordinates, frame=get_ROI_for_image(frame, coordinates)
            first_frame=1
            cv2.imshow("image",frame)
            k = cv2.waitKey(0)
            print "Press y if you want another crop or n if you dont"
            if k == ord("y"):
                further_ROIs==0
            else:
                further_ROIs==1
                break
            print "ROI section done. Creating full rectangles"
        cv2.destroyAllWindows()
        full_coordinates, path_to_save=construct_full_rectangles(coordinates)
        print "Total number of regions selected", len(full_coordinates)
        print "The coordinates have been saved in a clockwise order starting from top left"
        return path_to_save

            # futher_ROIs=1


        


    

    # video=cv2.VideoCapture(0)
    # if video.isOpened():
    #     rval, frame=video.read()
    # else:
    #     rval=False
#   while rval:



# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)

# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False

# while rval:
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#     key = cv2.waitKey(0)
#     if key == 0: # exit on ESC
#         break
# cv2.destroyWindow("preview")
# # It works in OpenCV-2.4.2 for me.

if __name__ =="__main__":
    path_to_save=capture_frames("out.mp4")