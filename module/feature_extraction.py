import cv2
from skimage.feature import hog, local_binary_pattern, hessian_matrix, hessian_matrix_eigvals, graycomatrix
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
import os
import argparse
from tqdm import tqdm 
from loguru import logger
from timeit import default_timer as timer
import random
from video_utils import split_feature_dataset
from scipy.ndimage import gaussian_filter

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

def get_flow(cubearr):
    '''
    Calculate optical flow between frames in a video.This function 
    takes a video file specified by 'videopath' and calculates the optical flow
    between consecutive frames using the Farneback method. 
    Optical flow represents the motion of objects in the video by estimating 
    the displacement of pixels between frames.

    Parameters:
    - cubearr (numpy.array): Numpy array which represents the frame cube to extract the features
    '''
    prvs = cv2.cvtColor(cubearr[0], cv2.COLOR_BGR2GRAY)

    # Create an empty list to store flow fields
    flow_fields = []

    for i in range(1, len(cubearr)):
        frame2 = cubearr[i]
        # Convert the current frame to grayscale
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 9, 1, 3, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # Store the flow field
        flow_fields.append(flow)
        # Update the previous frame
        prvs = next

    return flow_fields

def hof(flow, orientations, pixels_per_cell, cells_per_block, normalise, motion_threshold):
    '''
    Extract Histogram of Optical Flow (HOF) for a given flow field.

    Parameters:
    - flow: The optical flow field as a 2D NumPy array of shape (height, width, 2).
    - orientations: Number of orientation bins.
    - pixels_per_cell: Size (in pixels) of a cell.
    - cells_per_block: Number of cells in each block.
    - normalise: Apply power law compression to normalise the flow field if True.
    - motion_threshold: Threshold for considering motion.
    '''
    # Ensure that 'flow' is at least 2D
    flow = np.atleast_2d(flow)

    # Optionally apply normalization (sqrt)
    if normalise:
        flow = np.sqrt(flow)

    # Calculate the gradient in the x and y directions
    gx = flow[:, :, 1]
    gy = flow[:, :, 0]

    # Calculate the magnitude and orientation of the gradient
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    # Get the shape of the flow field
    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    # Calculate the number of cells in x and y directions
    n_cellsx = int(sx // cx)
    n_cellsy = int(sy // cy)

    # Initialize an orientation histogram
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))

    # Define a subsample for efficient calculation
    subsample = (slice(cy // 2, cy * n_cellsy, cy), slice(cx // 2, cx * n_cellsx, cx))

    # Iterate through orientation bins
    for i in range(orientations - 1):
        # Select pixels within the current orientation bin
        temp_ori = np.where((orientation >= 180 / orientations * i) & (orientation < 180 / orientations * (i + 1)), orientation, -1)
        # Consider only pixels with motion above the threshold
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)
        # Apply uniform filter to compute the histogram values for the cell
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[..., i] = temp_filt[subsample]

    # Handle the last orientation bin
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)
    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[..., -1] = temp_filt[subsample]

    # Calculate the number of blocks in x and y directions
    n_blocksx = int(n_cellsx - bx + 1)
    n_blocksy = int(n_cellsy - by + 1)

    # Initialize an array to store normalized blocks
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientations))

    # Normalize each block using L2-Hys
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / np.sqrt(block.sum()**2 + eps)

    return normalised_blocks.ravel()

def lbp(cubearr):
    '''
    Calculates the Local Binary Pattern (LBP) representation of a 3D cube array.
    The function uses a specified radius and number of points to compute the LBP patterns.

    Parameters:
    - cubearr (numpy.ndarray): Input 3D cube array, typically representing a video frame.
    '''

    radius = 5
    n_points = 16 * radius
    image = cubearr[...,0]

    image = gaussian_filter(image, sigma=2)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    return lbp.flatten()

def hessian_eigen(cubearr):
    '''
    ""
    Computes the eigenvalues of the Hessian matrix for a 3D cube array.

    Parameters:
    - cubearr (numpy.ndarray): Input 3D cube array, typically representing a video frame.
    '''

    # Compute the Hessian matrix
    H_elems = hessian_matrix(cubearr, sigma=0.2, order='rc', mode='reflect', cval=0.1)
    # Compute the eigenvalues
    eigenvalues = hessian_matrix_eigvals(H_elems)[0]

    return eigenvalues.flatten()

def glcm(cubearr):
    '''
    Computes the Gray-Level Co-occurrence Matrix (GLCM) for a 3D cube array.
    The function calculates the GLCM, capturing the spatial relationships of pixel intensities in the input cube array.
    It considers multiple angles to account for texture variations.

    Parameters:
    - cubearr (numpy.ndarray): Input 3D cube array, typically representing a video frame.
    '''

    # angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    glcm = graycomatrix(cubearr[...,0], distances=[5], angles = [0] , levels=256,
                        symmetric=False, normed=True)
    
    return glcm.flatten()

def calculate_features(cube, orientations, pixels_per_cell, cells_per_block, method):
    '''
    Calculates and extracts features from a frame cube.

    Parameters:
    - cube (str): Path to a frame cube from which to calculate the specified features.
    - orientations (int): The number of orientations for the feature calculation.
    - pixels_per_cell (int, int): The size of pixels in each cell for feature calculation.
    - cells_per_block (int, int): The number of cells in each block for feature calculation.
    - method (str): The method to be used for feature extraction: 'hog', 'hof', or 'hoghof'.
    '''

    logger.info("---Starting calculating features.")
    features = []
    gas_arr = read_video_to_array(cube)
   
    progress_bar = tqdm(total=len(gas_arr), desc=f"Processing {method[0].upper()}", unit="frame")
    print(method[0])
    if method[0] == 'hog':
        # Caclulation of Histogram Oriented Gradients
        for frame in range(gas_arr.shape[0]):
            hog_vector = hog(gas_arr[frame], orientations, pixels_per_cell, cells_per_block, block_norm='L2-Hys', visualize=False, channel_axis=-1)
            features.append(hog_vector)
            progress_bar.update(1)  
        
    elif method[0] == 'hof':
        # Caclulation of Histogram of Optical Flow
        flow_fields = get_flow(gas_arr)
        motion_threshold = .1 
        for flow in flow_fields:
            hof_vector = hof(flow, orientations, pixels_per_cell, cells_per_block, False, motion_threshold)
            features.append(hof_vector)
            progress_bar.update(1)
    
    elif method[0] == 'hoghof':
        hog_features, hof_features = [], []
        # Caclulation of Histogram Oriented Gradients
        for frame in range(gas_arr.shape[0]):
            hog_vector = hog(gas_arr[frame], orientations, pixels_per_cell, cells_per_block, block_norm='L2-Hys', visualize=False, channel_axis=-1)
            hog_features.append(hog_vector)
            progress_bar.update(1)  
        # Caclulation of Histogram of Optical Flow
        flow_fields = get_flow(gas_arr)
        motion_threshold = .1 
        for flow in flow_fields:
            hof_vector = hof(flow, orientations, pixels_per_cell, cells_per_block, False, motion_threshold)
            hof_features.append(hof_vector)
            progress_bar.update(1)
        features = hog_features + hof_features
    
    elif method[0] == 'lbp':
         # Caclulation of Local Binary Patterns
        for frame in range(gas_arr.shape[0]):
            lbp_vector = lbp(gas_arr[frame])
            features.append(lbp_vector)
            progress_bar.update(1)

    elif method[0] == 'hessian':
         # Caclulation of Hessian Eigenvalues
        for frame in range(gas_arr.shape[0]):
            hessian = hessian_eigen(gas_arr[frame])
            features.append(hessian)
            progress_bar.update(1)  

    elif method[0] == 'glcm':
        # Caclulation of Gray-Level Co-occurrence Matrix
        for frame in range(gas_arr.shape[0]):
            glcm_vector = glcm(gas_arr[frame])
            features.append(glcm_vector)
            progress_bar.update(1)
    else:
        raise ValueError("Invalid method. Supported methods: 'hog', 'hof', 'hoghof', 'lbp', 'hessian', 'glcm'")
    
    progress_bar.close() 
    features = np.array(features).astype(np.float16)

    return features

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate HOG-HOF features for video cubes.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the cube files folder (Required)")
    parser.add_argument("-i", "--index", type=int, help="Index of number of cubes to analyze")
    parser.add_argument("-o", "--orientations", type=int, default=9, help="Number of orientations (e.g., '9')")
    parser.add_argument("-ppc", "--pixels_per_cell", type=str, default="5,5", help="Pixels per cell (e.g., '5,5')")
    parser.add_argument("-cpb", "--cells_per_block", type=str, default="3,3", help="Cells per block (e.g., '3,3')")
    parser.add_argument("-m", "--method", choices=["hog", "hof", "hoghof", "lbp", "hessian", "glcm"], nargs="+", default=[], help="Choose what type features to calculate (Hog, Hof, HogHof, LBP, Hessian Eigenvalues or GLCM)")
    return parser.parse_args()

def main():
    time_script_start = timer()
    logger.info("--Feature extraction module")
    args = parse_args()
    cube_files = args.path

    # Check if the provided path exists
    if not os.path.exists(cube_files):
        logger.info("The specified folder does not exist.")
        return

    # List video cube files in the folder
    cubes = [str(x) for x in list(Path(cube_files).glob("*.mp4"))]
    random.shuffle(cubes)
    if not cubes:
        logger.info("No video cube files found in the specified folder.")
        return

    if args.index is not None:
        i = args.index
        a = range(i)
    else:
        # Process all available files when no index is provided
        a = range(len(cubes)) 

    # Parse pixels_per_cell and cells_per_block from strings to tuples
    pixels_per_cell = tuple(map(int, args.pixels_per_cell.split(",")))
    cells_per_block = tuple(map(int, args.cells_per_block.split(",")))
   
    for idx in a:
        # Check if the index is within a valid range
        if 0 <= idx < len(cubes):
            # Get the path to the cube video file
            cube_path = cubes[idx]
            # Calculate features for the specified cube using the chosen parameters
            features = calculate_features(cube_path, args.orientations, pixels_per_cell, cells_per_block, args.method)
            # Create the path for saving the feature files
            path = os.path.join('/', os.path.join(*cube_files.split('/', -1)[:-1]))
            save_folder = os.path.join(path, f"dataset_arrays_{args.method[0]}")

            # Create the save folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                
            # Define the filename for the saved feature file
            filename = cube_path.split('/', -1)[-1][:-4]  + '_' + args.method[0] + '_' + str(args.path).split('/',-1)[-3] + '.npz'
            # Save the calculated features to a compressed NumPy file
            np.savez_compressed(os.path.join(save_folder, filename), features)
            # Log a success message
            logger.info(f"Cube: {cube_path.split('/',-1)[-1]} completed successfully.")
        else:
            # Log a message if the index is not valid
            logger.info("Invalid cube index. Please provide a valid index.")

    num = 0.8
    split_feature_dataset(save_folder, num)

    time_script_end = timer()
    logger.debug(
        "---Execution time:\t%2.2f minutes" % np.round((time_script_end - time_script_start) / 60, 2))

if __name__ == "__main__":
    main()