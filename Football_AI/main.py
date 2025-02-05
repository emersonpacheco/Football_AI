from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from pitch_points import PitchPoints

def main ():

    video_frames = read_video('input_videos/input_video.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path='stubs/track_stub.pkl')

    tracker.add_position_to_tracks(tracks)

    pitch_pointer = PitchPoints('pitch_models/best.pt')

    points = pitch_pointer.get_points(video_frames,read_from_stub=True, stub_path='stubs/points.pkl')

    camera_movement = CameraMovementEstimator()
    camera_movement_per_frame = camera_movement.camera_mov_per_sec(points)

    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    begin_team_detect_frame = 0
    team_assigner = TeamAssigner()
    team_assigner.assing_team_color(video_frames[begin_team_detect_frame], tracks['players'][begin_team_detect_frame])
   
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()

    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if frame_num<=begin_team_detect_frame:
            continue
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

        else:
            if len(team_ball_control)==0:
                continue
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    output_video_frames = pitch_pointer.draw_points(points, output_video_frames)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__=='__main__':
    main()
