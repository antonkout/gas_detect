import cv2
import numpy as np
from timeit import default_timer as timer
from loguru import logger  
from dataset_creation import get_dimensions
from pathlib import Path
import random
import shutil
import os

def get_videometada(videopath):
   '''
   This function reads a video given a video path
   and provides info regarding the number of frames, the 
   width and height frames.

   Parameters:
      - videopath (str): Path to the video.
   '''

   cap = cv2.VideoCapture(videopath)

   # Get the total number of frames in the video
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

   # Get the frame width and height
   frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
   # Release the video capture object
   cap.release()

   # Close all OpenCV windows
   cv2.destroyAllWindows()

   return (total_frames, frame_height, frame_width, 3)

def get_positive_cubes(video):
    '''
    This function processes a video located at the specified input path 
    and detects the non-zero patches within the video frames. Then calculates
    the total number of the non-zero pixels and if the 80% of the total patch 
    is not covered by 0 values then converts the whole patch to ones.

    Parameters:
        - video (str): The path to the video file that will be processed.
    '''
    no_cols, no_rows = 5, 4
    cap = cv2.VideoCapture(video)

    dims =  get_videometada(video)

    fg_poscubes = np.zeros((dims), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        for blob_no in range(1, (no_cols * no_rows)+1):
            x_start, x_end, y_start, y_end = get_dimensions(frame, no_cols, no_rows, blob_no)
            arr = frame[y_start:y_end, x_start:x_end, :]
            count_ones = np.count_nonzero(arr)
            
            if count_ones > int((arr.shape[0] * arr.shape[1]) * 0.5):
                if frame_count < fg_poscubes.shape[0]:
                    fg_poscubes[frame_count, y_start:y_end, x_start:x_end, :] = 255
        
    cap.release()
    cv2.destroyAllWindows()

    return fg_poscubes

def videoframes_cubes(videopath, bgcubes):
    '''
    This function masks out the non-positive areas from
    the background extraction video.

    Parameters:
        - videopath (str). The original video containing the gas plume.
        - bgcubes (str). The positive cube video, contatining the positive foreground patches.
    '''
    new_frames = []
    cap1, cap2 = cv2.VideoCapture(videopath), cv2.VideoCapture(bgcubes)
    
    while True:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        if not _:
            break

        frame1[frame2==0] = 0
        new_frames.append(frame1)
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    return np.array(new_frames)

def read_video_to_array(video_path):
    '''
    Opens an MP4 video file and returns it as an array.

    Parameters:
    - video_path (str): Path to the input MP4 video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames)

def save_frames_from_video(video_path):
    '''
    Allows the user to select frames from a video and save them as images.

    Parameters:
    - video_path (str): Path to the input video file.
    '''
    frame_number = 1
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # cv2.imshow('Video Frame', frame)

        key = cv2.waitKey(30)

        if key == ord('s'):
            frame_filename = f'frame_{frame_number}.jpg'
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_number} saved as {frame_filename}")
            frame_number += 1

        if key == 27:  # 27 is the ASCII code for the 'Esc' key
            print("Video playback stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

def display_video_blobs(videopath, no_cols, no_rows, start_frame):
    '''
    Displays a video with a grid overlay, where each cell of the grid is numbered sequentially.
    The function allows pausing and resuming the video playback and provides cell numbering.

    Parameters:
    - videopath (str): Path to the input video file.
    - no_cols (int): Number of columns for the grid overlay.
    - no_rows (int): Number of rows for the grid overlay.
    - start_frame (int): The frame number from which to start displaying the video.
    '''
    cap = cv2.VideoCapture(videopath)
    
    frame_number = start_frame
    # Initialize the 'paused' variable
    paused = False  
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        if not paused:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

        # Calculate the width and height of each blob
        blob_width = frame.shape[1] // no_cols
        blob_height = frame.shape[0] // no_rows

        # Draw grid lines (red lines)
        for i in range(1, no_cols):
            cv2.line(frame, (i * blob_width, 0), (i * blob_width, frame.shape[0]), (0, 255, 0), 2)
        for i in range(1, no_rows):
            cv2.line(frame, (0, i * blob_height), (frame.shape[1], i * blob_height), (0, 255, 0), 2)

        # Draw blob IDs and cell numbers on the frame
        for row in range(no_rows):
            for col in range(no_cols):
                cell_number = row * no_cols + col + 1
                cell_x = col * blob_width
                cell_y = row * blob_height

                # Calculate text size to determine its width
                text = str(cell_number)
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Calculate text position to fit within the frame
                text_x = cell_x + (blob_width - text_width) // 2
                text_y = cell_y + 30

                # Draw cell number
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw frame number
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

        if not paused:
            frame_number += 1

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def save_video(outpath, data):
    '''
    This function saves at the provided outpath the data of video.

    Parameters:
        - outpath (str). Path to save the video
        - data (array). Numpy array that contains the frame values.
    '''

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, 25.0,
                            (data.shape[2], data.shape[1]), isColor=1)

    for frame in data:
        out.write(frame)
    out.release() 

def save_frames_localy(path, outpath):
    '''
    This function is used to capture and save individual frames from a video 
    located at the specified input path. It reads the video frame by frame, stores 
    the frames in a NumPy array, and then saves the frames as a compressed NumPy archive 
    (NPZ) file at the specified output path.

    Parameters:
        - path (str): The path to the input video file to capture frames from.
        - outpath (str): The directory where the NPZ file containing the captured frames will be saved.

    '''
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame.")
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    filename = path.split('/',-1)[-1].split('.',-1)[0] + "_frames.npz"
    np.savez_compressed(os.path.join(outpath,filename), frames)

def resize_video(input_video_path, output_video_path, new_width, new_height):
    '''
    Resizes a video from the input path to the specified dimensions and saves it to the output path.

    Parameters:
    - input_video_path (str): Path to the input video file.
    - output_video_path (str): Path to save the resized video.
    - new_width (int): The desired width of the resized video.
    - new_height (int): The desired height of the resized video.
    '''
    cap = cv2.VideoCapture(input_video_path)
    # Define the codec for the output video (in this case, MP4V)
    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    # Create VideoWriter object to write the resized video
    out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the new dimensions
        resized_frame = cv2.resize(frame, (new_width, new_height))
        # Write the resized frame to the output video
        out.write(resized_frame)
        # Break the loop when 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video objects and close OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def split_feature_dataset(feature_path, num):
    '''
    Splits generated feature arrays into training and test folders.
    Performs the following steps, creates 'training' and 'test' subdirectories within the provided 
    feature directory, lists all files in the feature directory and filters the files with filenames 
    containing 'gas' or 'non'. Then, shuffles the list of files for randomness and calculates a split 
    index (default 80% training, 20% test). Finally, copies files to the 'training' and 'test' directories
    based on the split index.
    
    Parameters:
        - feature_path (str): Path to features.
        - num (float): Percentage of training features (i.e. 0.8)
    '''

    # Create the training and test folders
    feature_path = Path(feature_path)

    # Create the training and test folders
    training_folder = feature_path / 'training'
    test_folder = feature_path / 'test'
    training_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    # List all files in the folder
    all_files = [f for f in feature_path.iterdir() if f.is_file()]
    # Filter files containing 'gas' or 'non' in the filename
    gas_non_files = [f for f in all_files if 'gas' in f.name or 'non' in f.name]

    # Shuffle the list of files for randomness
    random.shuffle(gas_non_files)

    # Calculate the split index
    split_index = int(num * len(gas_non_files))

    # Split the files into training and test sets
    training_files = gas_non_files[:split_index]
    test_files = gas_non_files[split_index:]

     # Copy files to the training and test folders
    for file in training_files:
        destination_path = training_folder / file.name
        shutil.copy(file, destination_path)

    for file in test_files:
        destination_path = test_folder / file.name
        shutil.copy(file, destination_path)

    logger.info(f"{len(training_files)} files copied to the training folder.")
    logger.info(f"{len(test_files)} files copied to the test folder.")

###############
# Usage example:
# input_video_path = '/home/antonkout/Documents/modules/flammable_gas_detection/release/data/gasmixture2/gasmixture2_r.wmv'
# output_video_path = '/home/antonkout/Documents/modules/flammable_gas_detection/release/data/gasmixture2/gasmixture2_sample.wmv'
# new_width = 640
# new_height = 512
# resize_video(input_video_path, output_video_path, new_width, new_height)

###############
# Example usage:
# video_path = '/home/antonkout/Documents/modules/flammable_gas_detection/dev/data/videos/propane_newdim.wmv'
# save_frames_from_video(video_path)

########
# videopath = '/home/antonkout/Documents/modules/flammable_gas_detection/release/data/methane/methane_sample.wmv'
# no_cols, no_rows = 5, 4
# start_frame = 0  
# display_video_blobs(videopath, no_cols, no_rows, start_frame)

#############
# videopath1 = "/home/antonkout/Documents/modules/flammable_gas_detection/release/data/gasmixture2/gasmixture2_sample.wmv"
# videopath2 = "/home/antonkout/Documents/modules/flammable_gas_detection/release/data//bg_squares.mp4"
# print(get_videometada(videopath1))