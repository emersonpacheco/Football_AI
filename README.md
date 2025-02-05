INSTRUCTIONS
    1.Train the player tracker model and put the trained model as 'best.pt' in the 'model' folder (I used the Roboflow dataset to train both models)
    2.Train the pitch points model and put as 'best.pt' in the 'pitch_models' folder
    3.Put your input video as 'input_video.mp4' in the 'input_videos' folder(be advise that longer videos may cause issues and/or take too long)
    4.Run main.py
    5. The 'stubs' folder cointains saved stubs for the player tracks and pitch points of the last run in order to make it faster to run multiple times for the same video and make changes, if you change the input video make sure to delete the stubs.
