from ultralytics import YOLO
import supervision as sv
import os
import cv2
import pickle
import sys
sys.path.append('../')
from football_pitch_config import SoccerPitchConfiguration

class PitchPoints:
    def __init__(self, pitch_points_model_path):
        self.model = YOLO(pitch_points_model_path)

    def get_points (self, video_frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                keypoints = pickle.load(f)
            return keypoints
        
        keypoints = []

        for frame in video_frames:
            results = self.model.predict(frame, conf=0.3)[0]
            frame_keypoints = sv.KeyPoints.from_ultralytics(results)
            keypoints.append(frame_keypoints)
            
        

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(keypoints, f)  

        return keypoints

    def draw_lines(self, annotated_frame,frame_num, keypoints):
        if len(keypoints[frame_num].xy)==0:
            return annotated_frame
        if keypoints[frame_num].xy[0][13].all()>0 and keypoints[frame_num].xy[0][14].all()>0:
            pt13 = tuple(map(int, keypoints[frame_num].xy[0][13]))
            pt14 = tuple(map(int, keypoints[frame_num].xy[0][14]))
            cv2.line(annotated_frame, pt13, pt14,(0,255,0),2)
        if keypoints[frame_num].xy[0][14].all()>0 and keypoints[frame_num].xy[0][15].all()>0:
            pt14 = tuple(map(int, keypoints[frame_num].xy[0][14]))
            pt15 = tuple(map(int, keypoints[frame_num].xy[0][15]))
            cv2.line(annotated_frame, pt14, pt15,(0,255,0),2)
        if keypoints[frame_num].xy[0][15].all()>0 and keypoints[frame_num].xy[0][16].all()>0:
            pt15 = tuple(map(int, keypoints[frame_num].xy[0][15]))
            pt16 = tuple(map(int, keypoints[frame_num].xy[0][16]))
            cv2.line(annotated_frame, pt15, pt16,(0,255,0),2)
        return annotated_frame

    def draw_points(self, keypoints, video_frames):
        CONFIG = SoccerPitchConfiguration()
        VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
        color=[sv.Color.from_hex(color) for color in CONFIG.colors],
        text_color=sv.Color.from_hex('#FFFFFF'),
        border_radius=5,
        text_thickness=1,
        text_scale=0.5,
        text_padding=5,
        )
        frames = []

        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints[frame_num], CONFIG.labels)
            annotated_frame = self.draw_lines(annotated_frame,frame_num,keypoints)
            frames.append(annotated_frame)
                
        return frames
        
 