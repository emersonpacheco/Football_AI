import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator:
    def __init__(self):
        pass
    def camera_mov_per_sec(self, keypoints):
        camera_movement = [[0,0]]*len(keypoints)
        
        for frame_num in range(1,len(keypoints)):
            camera_movement_frames = []
            
            if len(keypoints[frame_num].xy)==0 or len(keypoints[frame_num-1].xy)==0:
                continue
            for point, points in enumerate(keypoints[frame_num].xy[0]):
                prev_point = keypoints[frame_num-1].xy[0][point]
                curr_point = keypoints[frame_num].xy[0][point]
              
                if curr_point.all()>0 and prev_point.all()>0:
                    camera_movement_x, camera_movement_y = measure_xy_distance(prev_point, curr_point)
                    camera_movement_frames.append([camera_movement_x,camera_movement_y])
       
            camera_movement_array = np.array(camera_movement_frames)
            if len(camera_movement_array)==0:
               continue
            avg_x = np.mean(camera_movement_array[:,0])
            avg_y = np.mean(camera_movement_array[:,1])
            camera_movement[frame_num] =[avg_x,avg_y]
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (500,100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            if not x_movement > 1.00 or x_movement < -1.00:
                x_movement = 0.00
            if not y_movement > 1.00 or y_movement < -1.00:
                y_movement = 0.00

            frame = cv2.putText(frame, f"Camera Movement: x: {x_movement:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            frame = cv2.putText(frame, f"Camera Movement: y: {y_movement:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
            output_frames.append(frame)

        return output_frames