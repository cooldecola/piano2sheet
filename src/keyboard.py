from __future__ import unicode_literals
from usies_testing import cannyDetection, gaussianBlurring, keyboardYCoords
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import statistics
import mahotas       # Otsu thresholding
import bisect        # Key insert point
import imutils       # Crop and resize images
import pandas as pd

import youtube_dl
import os            # Folder paths
import sys           # Exit function
import glob          # Folder searching

from moviepy.editor import VideoFileClip  # Video processing - speeding up
import moviepy.video.fx.all as vfx


#importing files
from utils import getNextNote, keyboardYCoords, keyDetection, getConnectedComponentRectangle, timeNotes
from image_processing import *


class Keyboard:
    def __init__(self, youtubeURL, keyboard_array, white_width, black_width):
        self.youtubeURL = youtubeURL
        self.keyboard_array = keyboard_array
        self.white_width = white_width
        self.black_width = black_width

    # For downloading youtube videos
    def my_hook(self, d):
        if d['status'] == 'finished':
            print('Download complete.')
        elif d['status'] == 'error':
            print('Error in downloading file - exiting program!')
            sys.exit()

    # Clearing videos in video for file for next download
    def clear_previous(self, path="./videos/video_to_process*"):
        vtp = glob.glob(path)
        if vtp:
            os.remove(vtp[0])

    # Download youtube video

    def downloadYouTube(self, path='./videos/video_to_process.%(ext)s', quiet=True):
        #print(self)        # info for ydl_opts:
        # quiet:             Do not print messages to stdout.
        # outtmpl:           Template for output names.
        # progress_hooks:    A list of functions that get called on download
        #                progress, with a dictionary with the entries
        #                - status: One of "downloading", "error", or "finished".
        #                          Check this first and ignore unknown values.
        #                If status is one of "downloading", or "finished", the
        #                following properties may also be present:
        #                - filename: The final filename (always present)
        #                - tmpfilename: The filename we're currently writing to
        #                - downloaded_bytes: Bytes on disk
        #                - total_bytes: Size of the whole file, None if unknown
        #                - total_bytes_estimate: Guess of the eventual file size,
        #                                        None if unavailable.
        #                - elapsed: The number of seconds since download started.
        #                - eta: The estimated time in seconds, None if unknown
        #                - speed: The download speed in bytes/second, None if
        #                         unknown
        #                - fragment_index: The counter of the currently
        #                                  downloaded video fragment.
        #                - fragment_count: The number of fragments (= individual
        #                                  files that will be merged)
        #                Progress hooks are guaranteed to be called at least once
        #                (with status "finished") if the download is successful.
        self.clear_previous()
        ydl_opts = {'outtmpl': path,
                    'quiet': quiet,
                    'progress_hooks': [self.my_hook]}

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                print('Downloading video...')
                ydl.download([self.youtubeURL])
        except youtube_dl.utils.DownloadError:
            print('Exiting program!')

    # Use the file called video_to_process and detect our 88 keys
    def detect_keys(self, resize_width, bl_blur_sq, bl_canny_th1,
                    bl_canny_th2, bl_thresh1, bl_thresh2, wh_blur_sq,
                    path="./videos/video_to_process*", num_bl_keys=36,
                    num_wh_keys=52):

        # initializations
        frame_cnt = 0  # frame counter
        self.resize_width = resize_width
        vtp = glob.glob(path)
        tmp_ls_crop_coord = []  # list of keyboard Y coords

        # if file exists open camera
        if vtp:
            camera = cv2.VideoCapture(vtp[0])
            frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)  # total # of frames

            if not camera.isOpened():
                print('Error - video file could not be read!')
                sys.exit()
            else:
                print('Detecting keys...')
                while True:

                    (grabbed, frame) = camera.read()
                    if not grabbed:
                        break

                    frame = imutils.resize(
                        frame, width=resize_width)  # resizing

                    keys = frame
                    gray_keys = cv2.cvtColor(keys, cv2.COLOR_BGR2GRAY)

                    # processing
                    blurred = gaussianBlurring(gray_keys, blur_sq=bl_blur_sq)
                    edges = cannyDetection(
                        blurred, th1=bl_canny_th1, th2=bl_canny_th2)

                    # Send edged image to keyboardYCoords and use Hough line transform
                    # to obain coords that bound piano

                    # starting from first frame to halfway through the video we wanna
                    # see if they crop coordinate values stay the same. Then stop
                    # calculating and use the sam for the rest of the video
                    if (frame_cnt >= 0 and frame_cnt < int(0.5*frames)):
                        crop_coordinates = keyboardYCoords(edges)
                        self.y_coords = crop_coordinates
                        tmp_ls_crop_coord.append(crop_coordinates)

                    # after iterating through half the frames in video check if
                    # y coords have remained the same.. if so don't calculate
                    # y coords just use the previously established ones
                    if (frame_cnt >= int(0.5*frames)):
                        # checking if same elements in list
                        same = all(
                            element == tmp_ls_crop_coord[0] for element in tmp_ls_crop_coord)

                        if same:
                            crop_coordinates = tmp_ls_crop_coord[0]
                        else:
                            break

                    # cropping original image and gray image to get only the piano (bottom part)
                    cropped_keys = keys[int(
                        crop_coordinates[1])+20:int(crop_coordinates[0])]
                    cropped_gray_keys = gray_keys[int(
                        crop_coordinates[1])+20:int(crop_coordinates[0])]

                    # threshing
                    thresh_keys_bl = threshold(
                        cropped_gray_keys, th1=bl_thresh1, th2=bl_thresh2)

                    # labelling black keys
                    [num_labels_bl, labels_bl, stats_bl, centroids_bl] = connectedComponents(
                        binarized_img=thresh_keys_bl)  # info about black keys
                    final_labels_bl, key_width_bl = keyDetection(
                        cropped_keys, num_labels_bl, labels_bl, stats_bl, centroids_bl)
                    if len(final_labels_bl) == num_bl_keys:
                        first_note = 'A#0'
                        for i in range(num_bl_keys):
                            final_labels_bl[i][0] = first_note
                            first_note = getNextNote(first_note)

                        final_labels_bl = sorted(
                            final_labels_bl, key=lambda x: x[1])

                    # White keys
                    blurred_w = gaussianBlurring(cropped_gray_keys, blur_sq=wh_blur_sq)
                    T = mahotas.thresholding.otsu(blurred_w)*1.3
                    thresh_keys_w = cropped_gray_keys.copy()
                    thresh_keys_w[thresh_keys_w > T] = 255
                    thresh_keys_w[thresh_keys_w < T] = 0
                    [num_labels_w, labels_w, stats_w, centroids_w] = connectedComponents(binarized_img=thresh_keys_w)
                    final_labels_w, key_width_w = keyDetection(cropped_keys, num_labels_w, labels_w, stats_w, centroids_w)

                    if len(final_labels_w) == num_wh_keys:
                        first_note = "A0"
                        for j in range(num_wh_keys):
                            final_labels_w[j][0] = first_note
                            first_note = getNextNote(first_note)

                        final_labels_w = sorted(final_labels_w, key=lambda x: x[1])

                    # Determine if they sum to 88 keys (36 black, 52 white) - if not, try next frame
                    if len(final_labels_bl) == num_bl_keys and len(final_labels_w) == num_wh_keys:
                        self.black_keys = final_labels_bl
                        self.white_Keys = final_labels_w
                        self.black_width = key_width_bl
                        self.white_width = key_width_w
                        self.keyboard_img = frame
                        camera.release()  # Release cv2 camera object
                        cv2.destroyAllWindows()  # Destroy any cv2 windows
                        self.key_ranges()  # Call function to assign end-of-range for each key
                        print('Key detection complete.')
                        break

                    frame_cnt = frame_cnt + 1
            # If at this point we've looped through everything and we don't have 72 keys
            camera.release()
            cv2.destroyAllWindows()

            if len(final_labels_bl) != 36 and len(final_labels_w) != 52:
                print('Error in processing file - exiting program!')
                sys.exit()

        # Glob did not find a valid file
        else:
            print('Error - video path could not be accessed!')
            sys.exit()

    # Assign end of range for each key
    def key_ranges(self):
        # Array
        full_key_list = self.black_keys + self.white_Keys
        full_key_list = sorted(full_key_list, key=lambda x: x[1].astype(float))
        full_key_list = np.array(full_key_list)

        # Empty list
        tmp_list = np.empty([len(full_key_list), 2], dtype='object')

        # Loop through and assign end-of-range for each key
        # CASE: white key adjacent to black key: end of the white key range is the adjacent black key's centroid - black/2
        #       black key:                       end of the black key range is the black key's centroid + black/2
        #       white key adjacent to white key: end of the white key range is the half-way point between the adjacent centroids

        for i in range(0, len(full_key_list)-1):
            # White adjacent to black
            if len(full_key_list[i, 0]) == 1 and len(full_key_list[i+1, 0]) > 1:
                tmp_list[i, 1] = full_key_list[i+1,
                                               1].astype(float) - self.black_width/2
            elif len(full_key_list[i, 0]) > 1:  # Black Key
                tmp_list[i, 1] = full_key_list[i, 1].astype(
                    float) + self.black_width/2
            else:  # White key adjacent to white key
                tmp_list[i, 1] = (full_key_list[i, 1].astype(
                    float) + full_key_list[i+1, 1].astype(float))/2

            # No change to the actual note (only the distances, above) for the first key
            tmp_list[i, 0] = full_key_list[i, 0]

        # For the last key, just take it to infinity
        tmp_list[-1, 1] = np.inf
        tmp_list[-1, 0] = full_key_list[-1, 0]

        full_key_list = tmp_list
        self.keyboard_array = full_key_list

    def video_process(self, path="./videos/video_to_process*", offset_y_top=20, offset_y_bot=40, font=cv2.FONT_HERSHEY_SIMPLEX):
        #offset_y_top - how many pixels we crop from the top of each frame for detection
        # #offset_y_bot - how many pixels we crop from the bottom of the keyboard_array for detection
        #   Both of these ensure our detection only occurs in the middle of each frame - no effect on relative timing of notes

        offset_y_timing = offset_y_bot + 10
        # Our timing floor is offset from the detection floor by 10 pixels (timing is ABOVE detection)

        vtp = glob.glob(path)

        camera = cv2.VideoCapture(vtp[0])

        frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = camera.get(cv2.CAP_PROP_FPS)
        seconds_per_frame = fps/frames

        keys_timed_update = []
        for x in self.keyboard_array:
            keys_timed_update.append(x[0])

        while (camera.isOpened()):
            (grabbed, frame) = camera.read()

            # Calculate the elapsed time (in seconds)
            frame_number = camera.get(cv2.CAP_PROP_POS_FRAMES)
            elapsed = frame_number/fps

            if not grabbed:
                break

            # resize or else it won't work
            frame = imutils.resize(frame, width=self.resize_width)

            # Crop the top 20 pixels and bottom 50
            crop_frame = frame[offset_y_top:int(self.y_coords[1])-offset_y_bot]

            # threshold the cropped and grayed image
            crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            th_crop_frame = threshold(
                crop_frame_gray, thresh_type=cv2.THRESH_BINARY)

            # For each frame, obrain connected components
            [num_labels, labels, stats, centroids] = connectedComponents(
                th_crop_frame)

            output_img = frame.copy()

            i = 1
            # loop through all the found connected components
            while i < len(stats):

                # Width of i-th connected component
                curr_connected_comp = stats[i, cv2.CC_STAT_WIDTH]

                # !!! - Determine if the WIDTH is much bigger than the width of a white key and less than 3x? (So we don't get extraneous video text, etc.)
                if curr_connected_comp > self.white_width*1.25 and curr_connected_comp < self.white_width*3:

                    #Threshold just the large component of interest
                    componentMask = (labels == i).astype("uint8") * 255
                    # Replace this with video frame
                    threshMask = cv2.bitwise_and(
                        crop_frame_gray, crop_frame_gray, mask=componentMask)

                    # Histogram segregation of black/white key
                    # Grayscale has one channel so we use [0]
                    # Possible values range from 0 to 256
                    bin_scaler = 4
                    hist = cv2.calcHist([threshMask], [0], None, [
                                        256/bin_scaler], [1, 256])

                    # Use a histogram to compute the dominant non-black (i.e. not the background) colour. Use ~90% of this to threshold the image.
                    T = hist.argmax() * bin_scaler * .95
                    white_notes = threshMask.copy()
                    white_notes[white_notes > T] = 255
                    white_notes[white_notes < T] = 0

                    # Detect the first set of WHITE keys
                    [num_labels_wh, labels_wh, stats_wh,
                        centroids_wh] = connectedComponents(white_notes)

                    # Loop through components and determin which ones may be keys
                    for j in range(1, num_labels_wh):
                        area = stats_wh[j, cv2.CC_STAT_AREA]
                        # !!! - min white pixel area
                        # filtering out relavent detections (the ones big enough to be keys)
                        if (20 < area < np.inf):
                            if j > 1:
                                ## We've added another label
                                num_labels += 1
                                i += 1

                            #Within labels, we have a matrix that is the same size of the image that holds our split component
                            #First, cut out the original "fat" label - i.e. two or more keys are coupled together
                            coupled_keys_mask = labels != i
                            labels = labels * coupled_keys_mask

                            # Next, increment each lavel above the cut one up to accomodate the new label
                            higher_mask = labels > i
                            labels = labels + higher_mask

                            #Then append our segregated key
                            new_mask = labels_wh == j
                            new_labels = labels_wh * new_mask
                            new_labels = i * new_labels
                            labels = labels + new_labels

                            # Remove the original index for the stats and then add the new one
                            if i < len(stats):
                                stats = np.delete(stats, i, 0)
                                stats = np.insert(stats, i, stats_wh[j], 0)
                            elif j == 1:
                                stats = stats[:-1, :]
                                stats = np.concatenate(
                                    (stats, stats_wh[j][None, :]), 0)
                            else:
                                stats = np.concatenate(
                                    (stats, stats_wh[j][None, :]), 0)

                            # Remove the original index for the centroids and then add the new one
                            if i < len(centroids):
                                centroids = np.delete(centroids, i, 0)
                                centroids = np.insert(
                                    centroids, i, centroids_wh[j], 0)
                            elif j == 1:
                                centroids = centroids[:-1, :]
                                centroids = np.concatenate(
                                    (centroids, centroids_wh[j][None, :]), 0)
                            else:
                                centroids = np.concatenate(
                                    (centroids, centroids_wh[j][None, :]), 0)

                            # Plot immediately so indexing doesn't get messed up
                            # !!! - +20 pixels
                            [x, y, w, h, area] = getConnectedComponentRectangle(
                                stats[i, :])
                            y += offset_y_top  # We cropped out the first 20 pixels

                            # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                            if w >= self.black_width * 0.5:
                                # For each component, time it and draw via cv2
                                keys_timed_update = timeNotes(x, y, w, h, centroids[i], offset_y_top, h, self.y_coords[1]-offset_y_timing,
                                                              self.keyboard_array, keys_timed_update, elapsed, output_img, font=font, font_scale=0.5, font_color=(128, 0, 128))

                        #Detect the next set of keys
                        black_tmp = threshMask.copy()
                        # Invert the coupled image used for the white keys to ID black keys
                        black_tmp[black_tmp > T] = 0
                        # Need to blur to remove extraneous detail
                        blurred_black_notes = gaussianBlurring(
                            black_tmp, blur_sq=5)

                        # Using standard theshold segment just the black keys
                        black_notes = threshold(
                            blurred_black_notes, thresh_type=cv2.THRESH_BINARY)

                        # Detect the second set of keys
                        [num_labels_bl, labels_bl, stats_bl,
                            centroids_bl] = connectedComponents(black_notes)

                        #Loop through components and determine which ones may be keys
                        for k in range(1, num_labels_bl):
                            area = stats_bl[k, cv2.CC_STAT_AREA]

                            # !!! - min area
                            # filtering out relevant detection (the ones big enough to be keys)
                            if (20 < area < np.inf):
                                if k > 1:
                                    # We've added another label
                                    num_labels += 1
                                    i += 1

                                # For the second set of keys WE DON'T NEED TO CUT anything
                                # coupled_keys_mask = labels != i
                                # labels = labels * coupled_keys_mask

                                # Next, increment each label above the cut one up to accomodate the new label
                                higher_mask = labels > i + 1
                                labels = labels + higher_mask

                                #Then append our segregated key
                                new_mask = labels_bl == k
                                new_labels = labels_bl * new_mask
                                new_labels = (i + 1) * new_labels
                                labels = labels + new_labels

                                #Add
                                if i < len(stats):
                                    stats = np.insert(
                                        stats, (i+1), stats_bl[k], 0)
                                else:
                                    stats = np.concatenate(
                                        (stats, stats_bl[k][None, :]), 0)

                                # Added
                                if i < len(centroids):
                                    centroids = np.insert(
                                        centroids, (i+1), centroids_bl[k], 0)
                                else:
                                    centroids = np.concatenate(
                                        (centroids, centroids_bl[k][None, :]), 0)

                                # Plot immediately so indexing doesn't get messed up
                                # !!! - +20 pixels
                                [x, y, w, h, area] = getConnectedComponentRectangle(
                                    stats[i+1, :])
                                y += offset_y_top  # We cropped out the first 20 pixels

                                # !!! - minimum black width
                                # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                                if w >= self.black_width * 0.5:
                                    # For each component, time it and draw via cv2
                                    keys_timed_update = timeNotes(x, y, w, h, centroids[i+1], offset_y_top, h, self.y_coords[1]-offset_y_timing,
                                                                  self.keyboard_array, keys_timed_update, elapsed, output_img, font=font, font_scale=0.5, font_color=(0, 0, 255))

                else:
                    # !!! - +20 pixels
                    [x, y, w, h, area] = getConnectedComponentRectangle(
                        stats[i, :])
                    y += offset_y_top

                    # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                    # filtering out relavent detections (the ones big enough to be keys)
                    if (20 < area < np.inf) and w >= self.black_width * 0.5:
                        keys_timed_update = timeNotes(x, y, w, h, centroids[i], offset_y_top, h, self.y_coords[1]-offset_y_timing,
                                                      self.keyboard_array, keys_timed_update, elapsed, output_img, font=font, font_scale=0.33, font_color=(255, 255, 255))

                i += 1

            #Show frame + drawn rectangle
            cv2.imshow("Video", output_img)

            #Can break early by pressing "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) & 0xFF == ord("p"):
                cv2.waitKey(-1)  # wait until any key is pressed

        # print(keys_timed_update)
        camera.release()
        cv2.destroyAllWindows()
        self.timed_keys = keys_timed_update

    def export_notes(self):
        new_list = []

        for i in range(len(self.timed_keys)):
            temp = []
            temp.append(self.timed_keys[i][0])
            if (len(self.timed_keys[i]) > 1):
                for j in range(1, len(self.timed_keys[i])):
                    temp.append(self.timed_keys[i][j][0])

            new_list.append(temp)

        df = pd.DataFrame(new_list)
        df.to_csv('notes_info.csv', index=False, header=False)

    # Return the centroid of the black/white keys alongside the median width of each

    def getKeys(self):
        return [self.black_keys, self.white_keys, self.black_width, self.white_width]

    # Return the frame that 88 keys were successfully identified from
    def getFrame(self):
        return self.keyboard_img

    def getFullKeyList(self):
        return self.keyboard_array

    def getKeyboardYCoords(self):
        return self.y_coords

    def getTimedKeys(self):
        return self.timed_keys
