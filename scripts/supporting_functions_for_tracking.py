'''supporting_functions_for_tracking : Contains all supporting functions needed for tracking the displaced bin
   Author : Mohana Roy Chowdhury
   First edit : 18th July, 2017'''

import cv2
import numpy as np

def kl_tracking_initial_values():
    track_len=10
    detect_interval=100
    tracks=[]
    frame_idx=0 ##frame id ## can be removed later
    return track_len, detect_interval, tracks, frame_idx

def kl_feature_params_initializer():
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 10,
                           qualityLevel = 0.3,
                           minDistance = 1,
                           blockSize = 3 )
    return lk_params, feature_params
    
def convert_cropcoordis_fullimagecoordis(displayed_coordinates, x, y):
    x_full_image=x+displayed_coordinates[0]
    y_full_image=y+displayed_coordinates[1]
    return x_full_image, y_full_image



###p has the initial coordinate values and tracks have all of them appended in a list
def initializer_tracking_kl(object_intial_gray_crop, tracks, lk_params, feature_params, bin_roi):
    mask = np.zeros_like(object_intial_gray_crop)
    mask[:] = 255
    print "initial tracks", tracks
    # for x, y in [np.int32(tr[-1]) for tr in tracks]:
    #     cv2.circle(mask, (x, y), 5, 0, -1)
    p = cv2.goodFeaturesToTrack(object_intial_gray_crop, mask = mask, **feature_params) 
    
    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            x_full_image, y_full_image=convert_cropcoordis_fullimagecoordis(bin_roi[0], x, y)

            tracks.append([(x_full_image, y_full_image)])
    return tracks, p

# def tracker_kl_main():



# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# feature_params = dict( maxCorners = 500,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )


#     def __init__(self, video_src):
#         self.track_len = 10
#         self.detect_interval = 5
#         self.tracks = []
#         self.cam = video.create_capture(video_src)
#         self.frame_idx = 0

#     def run(self):
#         while True:
#             ret, frame = self.cam.read()
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             vis = frame.copy()

#             if len(self.tracks) > 0:
#                 img0, img1 = self.prev_gray, frame_gray
#                 p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
#                 p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
#                 p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
#                 d = abs(p0-p0r).reshape(-1, 2).max(-1)
#                 good = d < 1
#                 new_tracks = []
#                 for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
#                     if not good_flag:
#                         continue
#                     tr.append((x, y))
#                     if len(tr) > self.track_len:
#                         del tr[0]
#                     new_tracks.append(tr)
#                     cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
#                 self.tracks = new_tracks
#                 cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
#                 draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

#             if self.frame_idx % self.detect_interval == 0:
#                 mask = np.zeros_like(frame_gray)
#                 mask[:] = 255
#                 for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
#                     cv2.circle(mask, (x, y), 5, 0, -1)
#                 p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
#                 if p is not None:
#                     for x, y in np.float32(p).reshape(-1, 2):
#                         self.tracks.append([(x, y)])


#             self.frame_idx += 1
#             self.prev_gray = frame_gray
#             cv2.imshow('lk_track', vis)

#             ch = cv2.waitKey(1)
#             if ch == 27:
#                 break
