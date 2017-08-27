import cv2
import argparse

def construct_vid(video_path):
    img_path = "consolidated_tracking_frameset4/"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video= cv2.VideoWriter(video_path, fourcc, 5.0, (640,360))
    for i in range(1,10):
        frame= cv2.imread(img_path+"image_"+str(i)+".png")
        print frame.shape
        # frame = cv2.resize(frame, (640,480))
        video.write(frame)
        if i%1100==0:
            print i
    
    #print cut_thresh
    video.release()
    cv2.destroyAllWindows()

construct_vid("video1_test.avi")

# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)
#         out.write(frame)
   
#         # cv2.imshow('frame',frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#         # else:
#         #     break
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# # cut_off_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# #cut_off_threshold = [0.502]
# #for cut_thresh in cut_off_threshold:
# construct_vid('video1_test.avi')
# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
# #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()