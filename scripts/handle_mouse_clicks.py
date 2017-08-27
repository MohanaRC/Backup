'''handle_mouse_clicks.py : Allows the user to select the ROI using mouse clicks
   Author : Mohana Roy Chowdhury
   First edit : 18th July, 2017'''

import cv2
 
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
   Output Parameters :
   refPt : List holding the coordinates of the ROI as top left to bottom right'''

def get_ROI_for_image(image):
    clone=image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
    # print "Inside while loop and displaying the image"
    # display the image and wait for a keypress
        cv2.imshow("image", image)
        ### When two coordinates are obtained we draw a rectange
        if len(refPt)==2:
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)    
            print " type c to confirm r to reset"
        key = cv2.waitKey(1) & 0xFF    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            print "Pressing r, resetting the cropping region"
            image = clone.copy()
 
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            print "Pressing c and breaking off from the loop"
            break
    cv2.destroyAllWindows()
    return refPt

'''Note I am currently using it with another script that picks up video from frames and inputs the 
   first frame in the get_ROI_for_image function. One more way to make it more flexible would be to
   read video frames and use a keyboard input to skip the frames and on the desired frame the user
   can click and select ROI and then the rest of the video can be processed based on that'''

if __name__ =="__main__":
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread("frames_0001.png")
    print image.shape
    ROI_coordinates=get_ROI_for_image(image)
    print ROI_coordinates