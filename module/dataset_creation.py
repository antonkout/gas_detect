import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm 
from loguru import logger
from timeit import default_timer as timer
import json

def save_patches_dynamic(videopath, blob_no, start_frame, end_frame, save_folder, num_frames_to_extract, fps, name):
    '''
    Extracts a dynamic range of frames from a video, isolates a specified region of interest (ROI) or blob,
    and saves the extracted frames as a new video file. The function reads the video frames, extracts frames 
    within the specified frame range and isolates the specified ROI or blob. The resulting frames are saved as 
    a new video in the specified output folder.

    Parameters:
    - videopath (str): Path to the input video file.
    - blob_no (int): Number representing the specific ROI or blob to isolate within the video.
    - start_frame (int): The frame number from which to start extracting frames.
    - end_frame (int): The frame number at which to stop extracting frames.
    - save_folder (str): Path to the directory where the extracted video will be saved.
    - num_frames_to_extract (int): Number of frames to be extracted.
    - fps (int, optional): Frames per second of the output video.
    - name (str): Name of the cube (gas/non_gas)
    '''

    cap = cv2.VideoCapture(videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

    # Read all frames into an array
    frames_to_extract = []
    for _ in range(num_frames_to_extract):
        ret, frame = cap.read()
        if not ret:
            break
        frames_to_extract.append(frame)

    # Convert frames to a NumPy array
    frames_to_extract = np.array(frames_to_extract, dtype=np.uint8)

    # Extract the specified blob
    no_cols, no_rows = 5, 4  # Number of columns and rows for blob division
    x_start, x_end, y_start, y_end = get_dimensions(frames_to_extract[0], no_cols, no_rows, blob_no)
    cubic = frames_to_extract[:, y_start:y_end, x_start:x_end, :]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Create VideoWriter object to write the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cubename = f'cube_{start_frame}_{end_frame}_patch_{blob_no}_{name}.mp4'
    out = cv2.VideoWriter(os.path.join(save_folder, cubename), fourcc, float(fps),
                         (cubic.shape[2], cubic.shape[1]), isColor=1)

    # Write the frames to the output video
    progress_bar = tqdm(total=len(cubic), desc=f"Processing {cubename} Cube", unit="frame")
    for cube in cubic:
        out.write(cube)
        progress_bar.update(1)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def split_frames_into_cubes(margins, fps, num_frames_per_cube):
    '''
    Splits a list of frame margins into valid cubes, considering the given frames per second (fps)
    and the desired number of frames per cube.

    Parameters:
    - margins (list of tuples): A list of frame margins, where each tuple contains the starting frame,
      ending frame, and any additional data (e.g., blob numbers).
    - fps (int, optional): Frames per second of the video.
    - num_frames_per_cube (int, optional): The desired number of frames per cube.
    '''

    valid_cubes = []

    for margin in margins:
        start_frame, end_frame, *blobs = margin

        # Calculate the total number of frames within the margin
        total_frames = end_frame - start_frame + 1

        # Calculate the number of cubes that can be extracted with a step of 25 frames
        num_cubes = total_frames // fps

        # Iterate through the cubes
        for i in range(num_cubes):
            cube_start_frame = start_frame + i * fps
            cube_end_frame = cube_start_frame + num_frames_per_cube 

            # Ensure the cube doesn't exceed the end frame
            if cube_end_frame <= end_frame:
                valid_cubes.append((cube_start_frame, cube_end_frame, *blobs))

    return valid_cubes

def get_dimensions(frame, no_cols, no_rows, blob_no):
    '''
    Calculate the dimensions and coordinates of a specified region or blob within an image frame.
    This function calculates the dimensions and coordinates of a specified region or blob within
    an image frame. It divides the frame into a grid based on the given number of columns and rows,
    and then calculates the coordinates of the specified blob based on its unique identifier.

    Parameters:
    - frame (numpy.ndarray): Input image frame.
    - no_cols (int): Number of columns for dividing the frame into equal parts.
    - no_rows (int): Number of rows for dividing the frame into equal parts.
    - blob_no (int): Number representing the specific region or blob to calculate dimensions for.    
    '''

    # Calculate the width and height of each blob
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    blob_width = frame_width // no_cols
    blob_height = frame_height // no_rows
    
    # Extract the specified blob
    # Calculate the row and column indices of the specified blob_no
    blob_col = (blob_no - 1) % no_cols  # Calculate the column index (0-based)
    blob_row = (blob_no - 1) // no_cols  # Calculate the row index (0-based)

    # Calculate the starting and ending coordinates of the specified blob
    x_start = blob_col * blob_width
    x_end = (blob_col + 1) * blob_width
    y_start = blob_row * blob_height
    y_end = (blob_row + 1) * blob_height

    return x_start, x_end, y_start, y_end

def process_cubics(cubics, folder_name, args, name):
    '''
    Process a set of video cubics, isolate specified regions of interest (ROIs), and save them as video clips.
    The function splits the video frames into cubics, and saves them as separate video clips
    in the specified folder.

    Parameters:
    - cubics (list): List of video cubics to process.
    - folder_name (str): Name of the folder where the processed video clips will be saved.
    - args (Namespace): Namespace containing command-line arguments, including savepath, fps, and noframes.
    '''

    folder_path = os.path.join(args.savepath, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    frames_cubics = split_frames_into_cubes(cubics, args.fps, args.noframes)
    
    progress_bar = tqdm(total=len(frames_cubics), desc=f"Processing {folder_name} Cubes", unit="frame")
    
    for frame_set in frames_cubics:
        for i in range(2, len(frame_set)):
            save_patches_dynamic(args.videopath, frame_set[i], frame_set[0], frame_set[1], folder_path, args.noframes, args.fps, name)
            progress_bar.update(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset creation, splitting video to video cubes.")
    parser.add_argument("-v", "--videopath", type=str, required=True, help="Path to the video to be processed (Required)")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to output folder (Required)")
    parser.add_argument("-j", "--jsonpath", type=str, default='./cube_training_patches.json', help="Path to json of dataset definition (Required)")
    parser.add_argument("-f", "--fps", type=int, default=25, help="Define the fps number of video (e.g., '25')")
    parser.add_argument("-n", "--noframes", type=int, default=13, help="Select number of frames per cube (e.g., '25')")
    return parser.parse_args()

def main():
    time_script_start = timer()
    logger.info("--Dataset creation module")
    args = parse_args()
    
    # Open the json file that contains info for the training dataset
    with open(args.jsonpath, 'r') as file:
        data = json.load(file)

    # Access the dictionaries
    cubics_gas_present, cubics_gas_non_present = data['cubics_gas_present'], data['cubics_gas_non_present']
    
    # Process gas and non-gas cubics
    process_cubics(cubics_gas_present, 'cubes', args, 'gas')
    process_cubics(cubics_gas_non_present, 'cubes', args, 'nongas')

    time_script_end = timer()
    logger.debug(
        "---Execution time:\t%2.2f minutes" % np.round((time_script_end - time_script_start) / 60, 2))

if __name__ == "__main__":
    main()