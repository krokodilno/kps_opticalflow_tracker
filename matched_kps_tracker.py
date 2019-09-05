#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2 as cv
import os, sys
# import video
# from common import anorm2, draw_str
from time import clock

from helpers import white_balance, convolve_mask

MATCH_DIST_TH = 0.79
NUM_FRAMES_TO_PROCESS = 500
MIN_TRACK = 40
RANSAC_REPR_TH =  5.5
DETECT_INTERVAL = 100
WRITE_RESULT_FRAMES = False
(h, w) = (1080, 1920)


images_path = "../cut60_segmentation_frames/frames_04d"
masks_path =  "../cut60_segmentation_frames/masks_04d"
gif_path = "../animation/forwardback/"
texture_orig_path = '../aca95_texture_orig.png'
texture_adv_path = '../animation/forwardback/130.png'
out_frames = './demo_frames'


files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
files = files[:NUM_FRAMES_TO_PROCESS]

masks = [os.path.join(masks_path, p) for p in sorted(os.listdir(masks_path))]
masks = masks[:NUM_FRAMES_TO_PROCESS]

gifs = [os.path.join(gif_path, p) for p in sorted(os.listdir(gif_path))]
gif_iter = iter(gifs)

texture_orig = cv.imread(texture_orig_path)
texture_orig = cv.resize(texture_orig, (w, h))

texture_orig_gray = white_balance(texture_orig, return_gray=True)

texture_adv = cv.imread(texture_adv_path)
texture_adv = cv.resize(texture_adv, (w, h))

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

class PTFAdvert:
    def __init__(self, track_len=10, detect_interval=100):
        self.track_len = track_len
        self.detect_interval = detect_interval
        self.tracks_akaze = []
        self.tracks = []
        self.accum_Mt = None
        #self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        

    def match_and_track(self):
        i=0
        print(i)
        while i<(len(files)):
            #_ret, frame = self.cam.read()
            print(files[i])
           
            frame = cv.imread(files[i])
            frame_gray = white_balance(frame, return_gray=True)
            
            canvas = cv.imread(masks[i], 0)
            canvas_convolved = convolve_mask(canvas)
            
            advert_flag = None
            
            i=i+1
            
            vis = frame.copy()
            
            try:
                texture_adv_path = next(gif_iter)
                print(texture_adv_path, '*' * 100)
                texture_adv = cv.imread(texture_adv_path)
                texture_adv = cv.resize(texture_adv, (w, h))
            except StopIteration:
                continue
                

            if len(self.tracks_akaze) > MIN_TRACK:
                
                img0, img1 = self.prev_gray, frame_gray
                
                p0 = np.float32([tr1[-1] for tr1 in self.tracks_akaze]).reshape(-1, 1, 2)
                
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                
                # check distance after forward-back OF
                good = d < 1
                
                new_tracks_akaze = []
                
                old_pts = []
                new_pts = []
                
                gen = zip(self.tracks_akaze, p1.reshape(-1, 2), p0.reshape(-1, 2), good)
                
                for tr1, (x, y), (x0, y0), flag in gen:
                    
                    if not flag:
                        continue
                        
                    tr1.append((x, y))
                    new_pts.append((x, y))
                    old_pts.append((x0, y0))
                    
                    if len(tr1) > self.track_len:
                        del tr1[0]
                        
                    new_tracks_akaze.append(tr1)
                    cv.circle(vis, (x, y), 2, (0,255, 0), -1)
                
                Mt, _ = cv.findHomography(np.asarray(old_pts), np.asarray(new_pts), cv.RANSAC, RANSAC_REPR_TH)
                
                if self.accum_Mt is None:
                    self.accum_Mt = Mt    
                else:
                    self.accum_Mt = self.accum_Mt.dot(Mt)
                    
#                 warped_logo = cv.warpPerspective(
#                         src=texture_adv,
#                         M=self.accum_Mt.dot(np.linalg.inv(self.matching_homo)),
#                         dsize=frame_gray.shape[::-1],
#                         flags=cv.INTER_LANCZOS4
#                     )
                    
                warped_logo = cv.warpPerspective(
                        src=self.prev_warped_logo,
                        M=Mt,
                        dsize=frame_gray.shape[::-1],
                        flags=cv.INTER_LANCZOS4
                    )
                
                overlayed_frame = cv.bitwise_or(frame, warped_logo, mask=canvas)
                vis = overlayed_frame.copy()
                
                self.tracks_akaze = new_tracks_akaze
                cv.polylines(vis, [np.int32(tr1) for tr1 in self.tracks_akaze], False, (0, 255 , 0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks_akaze))
            
            if self.frame_idx % self.detect_interval == 0 or len(self.tracks_akaze) <= MIN_TRACK:
                
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                
                detector = cv.AKAZE_create()  
                bf = cv.BFMatcher(cv.NORM_HAMMING)
                    
                kp1, fts1 = detector.detectAndCompute(frame_gray, mask=canvas_convolved)
                kp2, fts2 = detector.detectAndCompute(texture_orig_gray, mask=None)
                
                raw_matches= bf.knnMatch(np.asarray(fts1), np.asarray(fts2), 2)
                print('raw matches', len(raw_matches))
                
                matches = []

                for match in raw_matches:
                    if len(match) != 2:
                        continue

                    match_flag = match[0].distance < match[1].distance * MATCH_DIST_TH
                    if match_flag:
                        matches.append((match[0].trainIdx, match[0].queryIdx))

                pts1 = np.float32([kp1[i].pt for (_, i) in matches])
                pts2 = np.float32([kp2[i].pt for (i, _) in matches])
            
                M, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.5)
                
                print('+'* 100)
                print(frame_gray.shape[::-1], len(matches))
                
                    
                warped_logo = cv.warpPerspective(
                        src=texture_adv,
                        M=np.linalg.inv(M),
                        dsize=frame_gray.shape[::-1],
                        flags=cv.INTER_LANCZOS4
                    )
 
                overlayed_frame = cv.bitwise_or(frame, warped_logo, mask=canvas)
                vis = overlayed_frame.copy()
                    
                final=pts1
                
                final2=np.around(final)

                final2=final2.astype(int)
                print(final2)
                final3=np.array([])
                
                for num in final2:
                    if num not in final3:
                        final3=np.append(final3,num,axis=0)
                
                for x, y in [np.int32(tr1[-1]) for tr1 in self.tracks_akaze]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                    
                if final is not None:
                    for x, y in np.float32(final).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                if pts1 is not None:
                    for x, y in np.float32(pts1).reshape(-1, 2):
                        self.tracks_akaze.append([(x, y)])
                
            self.frame_idx += 1
            self.prev_gray = frame_gray
            self.prev_warped_logo = warped_logo
            self.matching_homo = M.copy()
            self.accum_Mt = None
            
            if WRITE_RESULT_FRAMES:
                cv.imwrite(out_frames + '/%04d.jpg' % i, vis)
            cv.imshow('optflow_track', vis)
            

            ch = cv.waitKey(24)
            if ch == 27:
                break

def main():

    ptf = PTFAdvert(detect_interval=DETECT_INTERVAL)
    ptf.match_and_track()
    
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    
    main()
    cv.destroyAllWindows()
    # cam.release()