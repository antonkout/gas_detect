import cv2
import numpy as np
import argparse
from timeit import default_timer as timer
from loguru import logger  
from video_utils import get_positive_cubes
from video_utils import save_video
import os

def weighted_average(in_vid, a, start_frame):
    '''
    Perform selective motion detection on a video using weighted average method.
    Displays the video with selective motion detection using the Weighted Average method, which is sensitive to movement.
    It detects fast-moving objects by applying a Gaussian blur and background subtraction using the MOG2 model.
    The fast moving objects are then enlarged and masked out from the weighted average output, in order to capture
    only the gas plumes.
    Press 'q' to quit, and 'Space' to pause/unpause the video.

    Parameters:
    - in_vid (str): Path to the input video file.
    - a (float): Learning rate for background subtraction (smaller values make the background update slower).
    - start_frame (int): The desired starting frame (default is 0).
    '''

    # Open and play the video
    cap = cv2.VideoCapture(in_vid)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Create a background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=250, detectShadows = True)
    
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    background_frame = "/home/antonkout/Documents/modules/flammable_gas_detection/release/data/background_frame.jpg"
    ret, frame = cap.read()
    # frame = cv2.imread(background_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.float32(frame)

    # burn_in_frames = int(fps * 10)
    # for i in range(burn_in_frames):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     avg += gray_frame
    # avg /= burn_in_frames 

    paused = False

    # Write out the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create VideoWriter object to write the resized video
    video_directory = os.path.dirname(in_vid)
    filename_bg = os.path.join(video_directory, 'background_result.mp4')
    filename_fg = os.path.join(video_directory, 'foreground_result.mp4')
    out1 = cv2.VideoWriter(filename_bg, fourcc, 25.0, (frame.shape[1], frame.shape[0]), isColor=0)
    out2 = cv2.VideoWriter(filename_fg, fourcc, 25.0, (frame.shape[1], frame.shape[0]), isColor=0)

    while True:
        if not paused:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break
           
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            kernel_shape = (5, 5)
            kernel = np.ones(kernel_shape, np.uint8)
            
            # Apply Gaussian blurring to the frame
            gray_frame = cv2.GaussianBlur(gray_frame, kernel_shape, 0)
            avg = cv2.accumulateWeighted(gray_frame, avg, a)
            res = cv2.convertScaleAbs(avg)

            # Extract foreground
            diff = cv2.absdiff(gray_frame, res)
            _, thresholded_diff1 = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY) #cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            # thresholded_diff1 = cv2.morphologyEx(thresholded_diff1, cv2.MORPH_OPEN, kernel)

            # Apply the background subtractor to obtain the foreground mask
            fg_mask = bg_subtractor.apply(gray_frame)

            # Threshold the binary mask
            _, thresholded_diff2 = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
            thresholded_diff2 = cv2.morphologyEx(thresholded_diff2, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((51, 51), np.uint8)
            thresholded_diff2 = cv2.morphologyEx(thresholded_diff2, cv2.MORPH_DILATE, kernel)

            # Create a copy to save the new data
            thresholded_diff = np.copy(thresholded_diff1)
            # Mask the fast-moving areas from the slow ones (mask out vehicles to catch the plume)
            thresholded_diff[thresholded_diff2 > 0] = 0
            kernel = np.ones((3, 3), np.uint8)
            thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)

            masked_frame = np.copy(gray_frame)
            mask = (thresholded_diff == 0)
            masked_frame[mask] = 0 

            # Write the output
            out1.write(thresholded_diff)
            out2.write(masked_frame)

            # Stack the original frame, the subtraction result, and the thresholded difference side by side
            stacked_frame = cv2.hconcat([gray_frame, thresholded_diff1, thresholded_diff2, thresholded_diff])

            # Define titles for each window
            titles = ["Source video", "Selective motion detection", "Background subtraction (MOG2)", "Result"]

            # Split the stacked frame into individual regions for each window
            num_windows = len(titles)
            window_height = stacked_frame.shape[0]
            window_width = stacked_frame.shape[1] // num_windows

            # Initialize the x-coordinate for the current window
            x = 0

            # Iterate through each window
            for i in range(num_windows):
                # Create a region for the current window
                window_region = stacked_frame[:, x:x+window_width]
                
                # Add the title text to the window
                cv2.putText(
                    window_region,
                    titles[i],
                    (10, 30),  # Position of the text (adjust as needed)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # Font scale
                    (255, 255, 255),  # Text color (white)
                    2,  # Thickness of the text
                    cv2.LINE_AA  # Line type
                )

                # Move the x-coordinate to the start of the next window
                x += window_width

            desired_width = stacked_frame.shape[1] // 2
            desired_height = stacked_frame.shape[0] // 2
            resized_frame = cv2.resize(stacked_frame, (desired_width, desired_height))

            # Display the background and foreground
            cv2.imshow("Background Extraction", resized_frame)

        key = cv2.waitKey(80)

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        if not paused:
            start_frame += 1

    cv2.destroyAllWindows()
    cap.release()
    out1.release()
    out2.release()
    return filename_bg

def parse_args():
    parser = argparse.ArgumentParser(description="Background extraction, remove background from video through weighted average techniques.")
    parser.add_argument("-v", "--videopath", type=str, required=True, help="Path to the video to be processed (Required)")
    parser.add_argument("-a", "--alpha", type=float, default=0.001, help="Learning rate for background subtraction (smaller values make the background update slower)")
    parser.add_argument("-f", "--frame", type=int, default=0, help="The desired starting frame of the video (default is 0)")
    parser.add_argument("-p", "--prediction_areas", type=bool, default=True, help="Extract prediction areas from the generated background video")
    return parser.parse_args()

def main():
    time_script_start = timer()
    logger.info("--Background extraction module")
    args = parse_args()
    filename_bg = weighted_average(args.videopath, args.alpha, args.frame)
    
    video_directory = os.path.dirname(args.videopath)

    if args.prediction_areas == True:
        data = get_positive_cubes(filename_bg)
        bgoutpath = os.path.join(video_directory, 'bg_squares.mp4')
        save_video(bgoutpath, data)

    time_script_end = timer()
    logger.debug(
        "---Execution time:\t%2.2f minutes" % np.round((time_script_end - time_script_start) / 60, 2))

if __name__ == "__main__":
    main()