import numpy as np
from joblib import load
from pathlib import Path
import cv2
from dataset_creation import get_dimensions
from feature_extraction import get_flow, hof, local_binary_pattern
from skimage.feature import hog
import argparse
from scipy.ndimage import gaussian_filter

def calculate_hof(cube, orientations, pixels_per_cell, cells_per_block):
    '''
    Calculates the Histogram of Orientation of Flow (HOF) values.

    Parameters:
    - cube. np.array: The 13 frame cube on which to calculate the flow indicators.
    '''
    
    # Define empty lists to store results for visualization and comparison
    hof_features = []

    pixels_per_cell = pixels_per_cell.split(",")
    pixels_per_cell = int(pixels_per_cell[0]), int(pixels_per_cell[1])

    cells_per_block = cells_per_block.split(",")
    cells_per_block = int(cells_per_block[0]), int(cells_per_block[1])

    flow_fields = get_flow(cube)
    motion_threshold = .1 

    # Calculate HOF features for each flow field in the gas dataset
    for flow in flow_fields:
        hof_vector = hof(flow, orientations, pixels_per_cell, cells_per_block, False, motion_threshold)
        hof_features.append(hof_vector)
    
    hof_features = np.array(hof_features).astype(np.float16)

    return hof_features

def calculate_hog(cube, orientations, pixels_per_cell, cells_per_block):
    '''
    Calculates the Histogram of Gradients (HOG) values.

    Parameters:
    - cube. np.array: The 13 frame cube on which to calculate the flow indicators.
    '''
    hog_features = []
    pixels_per_cell = pixels_per_cell.split(",")
    pixels_per_cell = int(pixels_per_cell[0]), int(pixels_per_cell[1])

    cells_per_block = cells_per_block.split(",")
    cells_per_block = int(cells_per_block[0]), int(cells_per_block[1])

    for frame in range(cube.shape[0]):
        hog_vector = hog(cube[frame], orientations, pixels_per_cell, cells_per_block, block_norm='L2-Hys', visualize=False, channel_axis=-1)
        hog_features.append(hog_vector)

    hog_features = np.array(hog_features).astype(np.float16)
    return hog_features

def calculate_lbp(cube):
    '''
    Calculates the Local Binary Pattern (LBP) representation of a 3D cube array.
    The function uses a specified radius and number of points to compute the LBP patterns.

    Parameters:
    - cubearr (numpy.ndarray): Input 3D cube array, typically representing a video frame.
    '''
    lbp_features = []
    radius = 5
    n_points = 16 * radius

    for frame in range(cube.shape[0]):
        image = cube[frame][...,0]
        image = gaussian_filter(image, sigma=2)
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp = lbp.flatten()
        lbp_features.append(lbp)

    lbp_features = np.array(lbp_features).astype(np.float16)
    return lbp_features

def parse_args():
    parser = argparse.ArgumentParser(description="Detect flammamble gases at thermal videos.")
    parser.add_argument("-v", "--vidpath", type=str, required=True, help="Path to videos folder (Required)")
    parser.add_argument("-c", "--classifier", type=str, required=True, help="Classifier to use for prediction")
    parser.add_argument("-o", "--orientations", type=int, default=9, help="Number of orientations (e.g., '9')")
    parser.add_argument("-ppc", "--pixels_per_cell", type=str, default="5,5", help="Pixels per cell (e.g., '5,5')")
    parser.add_argument("-cpb", "--cells_per_block", type=str, default="3,3", help="Cells per block (e.g., '3,3')")
    return parser.parse_args()

def main():
    
    args = parse_args()
    folderpath = Path(args.vidpath)
    videopath = str(folderpath / f"{str(folderpath).split('/',-1)[-1]}_sample.wmv")
    videobg = str(folderpath / "bg_squares.mp4")
    videofg = str(folderpath / "background_result.mp4")
    videofg2 = str(folderpath / "foreground_result.mp4")
   
    # sgd_classifier_path = "/home/antonkout/Documents/modules/flammable_gas_detection/dev/classifier/classifier.joblib"
    clf = load(Path(args.classifier))

    no_cols, no_rows = 5, 4
    frame_count = 0
    frame_interval = 13
    frames_cube, bg_cube = [], []
    cap, capbg, capfg, capfg2 = cv2.VideoCapture(videopath),  cv2.VideoCapture(videobg),  cv2.VideoCapture(videofg), cv2.VideoCapture(videofg2)

    paused = False
    start_frame = 0

    rectangles = [{} for _ in range(no_cols * no_rows)]

    positive_detections = 0
    proba = 0
    while True:
        if not paused:
            ret, frame = cap.read()
            _, framebg = capbg.read()
            _, framefg = capfg.read()
            _, framefg2 = capfg2.read()
            if not ret:
                break
            
            isolated_area = np.zeros((512, 640, 3), dtype=np.uint8)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            capbg.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            capfg.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            capfg2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count += 1
            frames_cube.append(frame)
            bg_cube.append(framebg)

            if frame_count % frame_interval == 0:   
                arr , bg_arr =  np.array(frames_cube), np.asarray(bg_cube)

                for blob_no in range(1, (no_cols * no_rows)+1):
                    x_start, x_end, y_start, y_end = get_dimensions(frame, no_cols, no_rows, blob_no)
                    cubebg = bg_arr[:, y_start:y_end, x_start:x_end, :]
                    
                    if np.all(cubebg > 0):
                        cube = arr[:, y_start:y_end, x_start:x_end, :]
                        
                        if "_hof_" in args.classifier:
                            X_pred = calculate_hof(cube, args.orientations, args.pixels_per_cell, args.cells_per_block)

                        elif "_hog_" in args.classifier:
                            X_pred = calculate_hog(cube, args.orientations, args.pixels_per_cell, args.cells_per_block)
                        
                        elif "_hoghof_" in args.classifier:
                            hog = calculate_hog(cube, args.orientations, args.pixels_per_cell, args.cells_per_block)
                            hof = calculate_hof(cube, args.orientations, args.pixels_per_cell, args.cells_per_block)
                            X_pred = np.concatenate([hog, hof], axis=0)

                        elif "_lbp_" in args.classifier:
                            X_pred = calculate_lbp(cube)
                        
                        y_pred = clf.predict(X_pred.reshape(1, -1))[0]
                        proba = np.round(clf.predict_proba(X_pred.reshape(1, -1))[0][1] * 100 , 2)#The probability that this belongs to the positive class
                        
                        if y_pred == 1:
                            rectangles[blob_no - 1] = {
                                "coords": (x_start, y_start, x_end, y_end),
                                "duration": 13,
                                "value":1
                            }
                            positive_detections += 1
                        else:
                            rectangles[blob_no - 1] = {
                                "coords": (x_start, y_start, x_end, y_end),
                                "duration": 13,
                                "value":0
                            }
                        
                frames_cube, bg_cube = [], []

            for rect_info in rectangles:
                if rect_info and rect_info["duration"] > 0:
                    x_start, y_start, x_end, y_end = rect_info["coords"]
                    if rect_info["value"]==1:
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)

                        # Assuming framefg2 is a grayscale frame
                        nonzero_mask = framefg2[y_start:y_end, x_start:x_end] > 0  # Create a mask for nonzero values in the region
                        # Apply a colormap only to the positive detection area in the region
                        colormask = cv2.applyColorMap(framefg2[y_start:y_end, x_start:x_end], cv2.COLORMAP_PARULA) # 255 - 
                        colored_region = frame[y_start:y_end, x_start:x_end].copy()
                        # Copy the colormap values to the corresponding locations in the original frame in the region
                        colored_region[nonzero_mask] = colormask[nonzero_mask]
                        # Update the original frame with the colored region
                        #####
                        # frame[y_start:y_end, x_start:x_end] = colored_region

                        isocolored = isolated_area[y_start:y_end, x_start:x_end].copy()
                        isocolored[nonzero_mask] = colormask[nonzero_mask]
                        isolated_area[y_start:y_end, x_start:x_end] = isocolored

                        percentage = np.round((np.sum(nonzero_mask[...,0])/(nonzero_mask.shape[0]*nonzero_mask.shape[1])*100),2)
                        labelper = f"{percentage}%" 
                        
                        label_position = (x_start + 45, y_start + 60)  # Adjust the position as needed
                        categ_position = (x_start + 40, y_start + 30)
                        if (percentage > 25 and percentage <= 50):
                            cv2.putText(isolated_area, labelper, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(isolated_area, 'Low', categ_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        elif (percentage > 50 and percentage <= 75):
                            cv2.putText(isolated_area, labelper, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 255), 2)
                            cv2.putText(isolated_area, 'Medium', categ_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 255), 2)
                        elif (percentage > 75):
                            cv2.putText(isolated_area, labelper, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(isolated_area, 'High', categ_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.putText(isolated_area, labelper, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(isolated_area, 'Very low', categ_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        rect_info["duration"] -= 1
                    
                    else:
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 3)

            cv2.putText(frame, f"Detections: {positive_detections}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2)
            cv2.putText(frame, f"Probability: {proba}%", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            stacked_frame = cv2.hconcat([frame, isolated_area, framefg, framebg])
            titles = ["Detection System", "Gas Presence" ,"Background Extraction", "Prediction Cubes"]

            # Split the stacked frame into individual regions for each window
            num_windows = len(titles)
            # window_height = stacked_frame.shape[0]
            window_width = stacked_frame.shape[1] // num_windows

            # Initialize the x-coordinate for the current window
            x = 0

            # Iterate through each window
            for i in range(num_windows):
                # Create a region for the current window
                window_region = stacked_frame[:, x:x+window_width]
                # Define the background color (black in BGR format)
                background_color = (0, 0, 0)

                # Define the position and size of the black background rectangle
                text_x, text_y = 0, 0  # Position of the text (adjust as needed)
                text_w, text_h = 800, 40  # Width and height of the background (adjust as needed)

                # Draw the black background rectangle
                cv2.rectangle(
                    window_region,
                    (text_x, text_y),
                    (text_x + text_w, text_y + text_h),
                    background_color,
                    thickness=cv2.FILLED  # Filled rectangle
                )
                # Add the title text to the window
                cv2.putText(
                    window_region,
                    titles[i],
                    (200, 30),  # Position of the text (adjust as needed)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # Font scale
                    (250, 250, 250), 
                    2,  # Thickness of the text
                    cv2.LINE_AA  # Line type
                )

                # Move the x-coordinate to the start of the next window
                x += window_width

            desired_width = stacked_frame.shape[1] // 2 + 100
            desired_height = stacked_frame.shape[0] // 2  + 50
            resized_frame = cv2.resize(stacked_frame, (desired_width, desired_height))
            cv2.imshow("Gas Detection", resized_frame)

            key = cv2.waitKey(160)

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            if not paused:
                start_frame += 1

    cap.release()
    capfg.release()
    capfg2.release()
    capbg.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()