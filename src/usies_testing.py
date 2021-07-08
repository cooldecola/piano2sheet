from __future__ import unicode_literals
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


# note class
class Note: 
    def __init__(self, centroid_x, y_dot):
        self.centroid_x = centroid_x
        self.y_dot = y_dot

# just  a function for printing images
def display_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
      
# function for getting next note
def getNextNote(first_note):
    if "#" in first_note: 
        octave = first_note[2]
        if first_note[:2] == "A#":
            return ("C#" + octave)
        elif first_note[:2] == "C#":
            return ("D#" + octave)
        elif first_note[:2] == "D#":
            return ("F#" + octave)
        elif first_note[:2] == "F#":
            return ("G#" + octave)
        elif first_note[:2] == "G#":
            next_octave = int(octave) + 1
            return ("A#" + str(next_octave))
    
    else: 
        octave = first_note[1]
        if first_note[0] == "A":
            return ("B" + octave)
        elif first_note[0] == "B":
            return ("C" + octave)
        elif first_note[0] == "C":
            return ("D" + octave)
        elif first_note[0] == "D":
            return ("E" + octave)
        elif first_note[0] == "E":
            return ("F" + octave)
        elif first_note[0] == "F":
            return ("G" + octave)
        elif first_note[0] == "G":
            next_octave = int(octave) + 1
            return ("A" + str(next_octave))

# Function for Gaussian Blurring
def gaussianBlurring(gray_img, blur_sq, std_dev = 0):
    return cv2.GaussianBlur(gray_img, (blur_sq, blur_sq), std_dev)

# Function for Canny Edge Detection
def cannyDetection(blurred_img, th1, th2, apertureSize = 3):
    return cv2.Canny(blurred_img, th1, th2, apertureSize)

# Function to return the top and bottom of the keyboard using HoughLines
def keyboardYCoords(edged_img, rho = 1, theta = np.pi/180, threshold = None):
    
    if threshold is None:
        threshold = edged_img.shape[1]//2 # half the width
    
    lines = cv2.HoughLines(edged_img, rho, theta, threshold) 
    y_cord = [] #the y-value of the lines generated from hough transform

    #iterating through lines
    for line in lines: 
        rho_l, theta_l = line[0]
        a = np.cos(theta_l)
        b = np.sin(theta_l)
        x0 = a * rho_l
        y0 = b * rho_l
        y_cord.append(y0) #appending to list

    y_cord.sort(reverse=True)
    return y_cord[0:2]

# Function to threshold an image
def threshold(gray_img, th1 = 90, th2 = 150, thresh_type = cv2.THRESH_BINARY_INV):
    _, threshed_img = cv2.threshold(gray_img, th1, th2, thresh_type)
    return threshed_img


        
# Function for doing connected components
def connectedComponents(binarized_img, connectivity = 8, ltype = cv2.CV_32S):
    output = cv2.connectedComponentsWithStats(binarized_img, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    return [num_labels, labels, stats, centroids]


# Function to calculate the centroids of detected connected components
#   This should be used for the 36 black keys and 52 white keys
#   use display_result = True to display the results on the source image
def keyDetection(orig_img, num_labels, labels, stats, centroids, min_key_area = 100, display_result = False):
    final_labels = []

    output_img = orig_img.copy()

    # Loop through the detected connected components
    for i in range(1, num_labels):
        [x,y,w,h,area] = getConnectedComponentRectangle(stats[i,:])
        (cX, cY) = centroids[i]
        
        # Consider only connected that have an area greater than min_key_area pixels
        if (min_key_area < area < np.inf):
            final_labels.append([i,cX])
            
            # Display on original image
            if (display_result):
                cv2.rectangle(output_img, (x,y), (x+w, y+h), (255,0,0),1)
                cv2.circle(output_img, (int(cX), int(cY)), 4, (255,255,0), -1)
#                 componentMask = (labels == i).astype("uint8") * 255
                cv2.imshow("Output", output_img)
                cv2.waitKey(0)

    key_width = statistics.median(stats[:, cv2.CC_STAT_WIDTH])
    cv2.destroyAllWindows()
    
    #Return the centroid of the processed connected components and the median width
    return final_labels, key_width

# Get the x,y coordinates and width + height of cv2.connectedcomponents output
#    Given the i-th components of the stats matrix (an N x 5 matrix)
def getConnectedComponentRectangle(ith_stats):
        x = ith_stats[cv2.CC_STAT_LEFT]
        y = ith_stats[cv2.CC_STAT_TOP]
        w = ith_stats[cv2.CC_STAT_WIDTH]
        h = ith_stats[cv2.CC_STAT_HEIGHT]
        area = ith_stats[cv2.CC_STAT_AREA]
        
        return [x,y,w,h,area]


def displayCentroid(key_list, img):
    y = img.shape[0]*3//4
    for (note, centroid) in key_list: 
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.line(img,(int(centroid),0),(int(centroid),900),(0,0,255),1)
        text_label = cv2.putText(img, note, (int(centroid), y), font, 0.5, (0,255,0), 1)
        cv2.imshow("Key Label", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Given an x-coordinate, return the appropriate insertion point in the keyboard
def key_pressed(key_list, key_index):
    insertion_point = bisect.bisect_left(key_list[:,1].astype(float),key_index)
    
    #Insertion outside our index, means to insert it at the end (return the last key)
    if insertion_point >= len(key_list):
        insertion_point = len(key_list)-1
#     print(insertion_point)
#     print('You pressed the {} key.'.format(key_list[insertion_point,0]))

    note = key_list[insertion_point,0]
    index = insertion_point

    return note, index





def timeNotes(x,y,w,h, ith_centroids, y_offset, note_height, keyboard_line, keyboard_array, keys_timed_update, 
              elapsed, output_img, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.5, font_color = (0,0,255)):
    # x,y,w,h - respective x-coord, y-coord, width and height of a connected component
    # ith_centroids - the centroid of a given note (i.e. an x,y pair)
    # y_offset - if the keyboard has been offset in the y direction (this allows the detection to stay out of the top/bottom edges)
    # note_height - from connectedcomponents
    # keyboard_line - the y-coordinate that serves as the threshold/line to calculate press down/up
    # keyboard_array - the matrix of keyboard notes/octaves and x-coordinates
    # key_timed_update - our array that holds on/off button for each note in our keyboard
    # elapsed - time in seconds
    # output_img - the screen we want to display to

    lag = 2 # number of pixels about the keyboard_line for detection
    
    # Get centroid of detected note
    (cX, cY) = ith_centroids
    cY = cY + y_offset # We cropped out the first 20 pixels

    # Y-coordinate of top and bottom edges of a note (this is truncated by the "size" of the detection area)
    dist_to_edge = note_height/2 #getting the distance from centroid to bottom edge for better detection later on
    top_dot = cY-dist_to_edge
    bottom_dot = cY+dist_to_edge

#     note = Note(cX, cY+dist_to_edge) #creating note object and adding to list

    if ( (int(bottom_dot) >= int(keyboard_line - lag)) and (int(bottom_dot) <= int(keyboard_line)) ):
        note_played, index = key_pressed(keyboard_array, cX)
        keys_timed_update[index].append([elapsed])
#         print(note_played)
#         print('='*50)

    if ( (int(top_dot) >= int(keyboard_line - lag)) and (int(top_dot) <= int(keyboard_line)) ):
        note_played, index = key_pressed(keyboard_array, cX)
        keys_timed_update[index].append([elapsed])

    note_played, _ = key_pressed(keyboard_array, cX)
    cv2.line(output_img, (0, int(keyboard_line)), (600, int(keyboard_line)), (0,0,255), 2)
    cv2.rectangle(output_img, (x,y), (x+w, y+h), font_color,1)
    cv2.circle(output_img, (int(cX), int(bottom_dot)), 1, (0,122,255), 3)
    cv2.circle(output_img, (int(cX), int(top_dot)), 1, (0,122,255), 3)
    cv2.putText(output_img, note_played, (int(cX), int(cY+dist_to_edge)), font, font_scale, font_color, 1)
    
    return keys_timed_update


class Keyboard: 
    def __init__(self, youtubeURL):
        self.youtubeURL = youtubeURL
        
    # For downloading YouTube videos
    def my_hook(self, d):
        if d['status'] == 'finished':
            print('Download complete.')
        elif d['status'] == 'error':
            print('Error in downloading file - exiting program!')
            sys.exit()
            
    # Remove previously downloaded file
    def clear_previous(self, path = "./videos/video_to_process*"):
        vtp = glob.glob(path) # glob checks if theres a file called video_to_process in videos file
        if vtp:
            #camera = cv2.VideoCapture(vtp[0])
            #camera.release()
            os.remove(vtp[0])

    # Function to download a YouTube video
    def downloadYouTube(self, path = './videos/video_to_process.%(ext)s', quiet = True):

        # info for ydl_opts:
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
            
    # Speed up a video by a factor by removing frames
    def speed_video(self, spd_factor = 5, path = "./videos/video_to_process*"):
        vtp = glob.glob(path)
        
        if vtp:
            
            # 1) Speed up the video, save as a copy
            vtp = vtp[0]
            file_ext_idx = vtp.find('.',1)
            tmp_vtp = vtp[:file_ext_idx] +"_COPY" + vtp[file_ext_idx:]
            
            in_loc = vtp
            out_loc = tmp_vtp

            # Import video clip
            clip = VideoFileClip(in_loc)
           # print("fps: {}".format(clip.fps))

            # Modify the FPS
            #clip = clip.set_fps(clip.fps * spd_factor)
            clip = clip.set_fps(clip.fps * 1)

            # Apply speed up
            final = clip.fx(vfx.speedx, spd_factor)
            #print("fps: {}".format(final.fps))

            # Save video clip
            final.write_videofile(out_loc)
            clip.close()

            # 2) Delete the original 1x speed video
            #camera = cv2.VideoCapture(vtp)
            #camera.release()
            os.remove(vtp)
            
            # 3) Rename the sped-up video to the original
            old_file_name = tmp_vtp
            new_file_name = vtp
            os.rename(old_file_name, new_file_name)
            
    # Use the file called video_to_process and detect our 88 keys
    def detect_keys(self, resize_width, bl_blur_sq, bl_canny_th1, bl_canny_th2, bl_thresh1, bl_thresh2,
                wh_blur_sq, path = "./videos/video_to_process*",num_bl_keys = 36, num_wh_keys = 52):
        
        self.resize_width = resize_width
        
        vtp = glob.glob(path)
        if vtp:
            camera = cv2.VideoCapture(vtp[0])
            # Could not access the file
            if not camera.isOpened():
                print('Error - video file could not be read!')
                sys.exit()
            # Process video
            else:
                print('Detecting keys...')
                while True:
                    #grabbed is a boolean than tells us if there is a valid frame
                    (grabbed, frame) = camera.read()
                    if not grabbed:
                        break
                    
                    #! UNCOMMENT IT LATER
                    #frame = imutils.resize(frame,width = resize_width)

                    # Get the bottom-half of the frame (where the keyboard lies) and process from here
                    # keys = frame[frame.shape[0]//2:,:]
                    # gray_keys = cv2.cvtColor(keys, cv2.COLOR_BGR2GRAY)
                    keys = frame
                    #print(keys)
                    gray_keys = cv2.cvtColor(keys, cv2.COLOR_BGR2GRAY)

                    # Process
                    blurred = gaussianBlurring(gray_keys, blur_sq = bl_blur_sq)
                    edges = cannyDetection(blurred, th1 = bl_canny_th1, th2 = bl_canny_th2)

                    # Crop
                    crop_coordinates  = keyboardYCoords(edges)
                    self.y_coords = crop_coordinates
                    cropped_keys      = keys[int(crop_coordinates[1])+20:int(crop_coordinates[0])]
                    cropped_gray_keys = gray_keys[int(crop_coordinates[1])+20:int(crop_coordinates[0])]

                    # Labels keys
                    thresh_keys_bl = threshold(cropped_gray_keys, th1 = bl_thresh1, th2 = bl_thresh2)

                    # Black keys
                    [num_labels_bl, labels_bl, stats_bl, centroids_bl] = connectedComponents(binarized_img = thresh_keys_bl)
                    final_labels_bl, key_width_bl = keyDetection(cropped_keys, num_labels_bl, labels_bl, stats_bl, centroids_bl)
                    if len(final_labels_bl) == num_bl_keys: 
                        first_note = "A#0"
                        for i in range(num_bl_keys):
                            final_labels_bl[i][0] = first_note
                            first_note = getNextNote(first_note)

                        final_labels_bl = sorted(final_labels_bl, key=lambda x: x[1])

                    # White keys
                    blurred_w = gaussianBlurring(cropped_gray_keys, blur_sq = wh_blur_sq)
                    T = mahotas.thresholding.otsu(blurred_w)*1.3
                    thresh_keys_w = cropped_gray_keys.copy()
                    thresh_keys_w[thresh_keys_w>T] = 255
                    thresh_keys_w[thresh_keys_w<T] = 0
                    [num_labels_w, labels_w, stats_w, centroids_w] = connectedComponents(binarized_img = thresh_keys_w)
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
                        self.white_keys = final_labels_w
                        self.black_width = key_width_bl
                        self.white_width = key_width_w
                        self.keyboard_img = frame
                        camera.release()         # Release cv2 camera object
                        cv2.destroyAllWindows()  # Destroy any cv2 windows
                        self.key_ranges()             # Call function to assign end-of-range for each key
                        print('Key detection complete.')
                        break
                    #Show the frame + drawn rectangle
                    # cv2.imshow("Face", thresh_keys_bl)
                    # #Can break early by pressing "q"
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    # break

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
            
            
            
    # Assign end-of-range for each key
    def key_ranges(self):

        # Array
        full_key_list = self.black_keys + self.white_keys
        full_key_list = sorted(full_key_list, key=lambda x: x[1].astype(float))
        full_key_list = np.array(full_key_list)

        # Empty list
        tmp_list = np.empty([len(full_key_list), 2], dtype='object')

        # Loop through and assign end-of-range for each key
        # CASE: white key adjacent to black key: end of the white key range is the adjacent black key's centroid - black/2
        #       black key:                       end of the black key range is the black key's centroid + black/2
        #       white key adjacent to white key: end of the white key range is the half-way point between the adjacent centroids
        for i in range(0,len(full_key_list)-1):
            if len(full_key_list[i,0])==1 and len(full_key_list[i+1,0])>1: # White adjacent to black
                tmp_list[i,1] = full_key_list[i+1,1].astype(float) - self.black_width/2
            elif len(full_key_list[i,0])>1: # Black key
                tmp_list[i,1] = full_key_list[i,1].astype(float) + self.black_width/2
            else: # White key adjacent to white key
                tmp_list[i,1] = (full_key_list[i,1].astype(float)+ full_key_list[i+1,1].astype(float))/2

            # No change to the actual note (only the distances, above) for the first key
            tmp_list[i,0] = full_key_list[i,0]

        #For the last key, just take it to infinity
        tmp_list[-1,1] = np.inf
        tmp_list[-1,0] = full_key_list[-1,0]

        full_key_list = tmp_list

        self.keyboard_array = full_key_list
        
    ################################################################################
    def video_process(self, path = "./videos/video_to_process*", offset_y_top = 20, offset_y_bot = 40,font = cv2.FONT_HERSHEY_SIMPLEX):
     
        #offset_y_top - how many pixels we crop from the top of each frame for detection
        #offset_y_bot - how many pixels we crop from the bottom of the keyboard_array for detection
        #   Both of these ensure our detection only occurs in the middle of each frame - no effect on relative timing of notes
        
        offset_y_timing = offset_y_bot + 10    
        #   Our timing floor is offset from the detection floor by 10 pixels (timing is ABOVE detection)
    
        vtp = glob.glob(path)
        
        camera = cv2.VideoCapture(vtp[0])

        frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = camera.get(cv2.CAP_PROP_FPS)
        seconds_per_frame = fps/frames

        

        keys_timed_update = []
        for x in self.keyboard_array:
            keys_timed_update.append([x[0]])

        while (camera.isOpened()):    
            #grabbed is a boolean than tells us if there is a valid frame
            (grabbed, frame) = camera.read()

            # Calculate the elapsed time (in seconds)
            frame_number = camera.get(cv2.CAP_PROP_POS_FRAMES)
            elapsed = frame_number/fps

            if not grabbed:
                break       

            frame = imutils.resize(frame,width = self.resize_width) #resize or else it won't work

            # Our detection range is from: Top of frame + offset_y_top
            #                          to: Top of keyboard - offset_y_bot
            crop_frame = frame[offset_y_top:int(self.y_coords[1])-offset_y_bot] #Crop the top 20 pixels and bottom 50

            # threshold the cropped and grayed image
            crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            th_crop_frame = threshold(crop_frame_gray, thresh_type = cv2.THRESH_BINARY)

            # For each frame, obtain connected components
            [num_labels, labels, stats, centroids] = connectedComponents(th_crop_frame)

            output_img = frame.copy()
            

            i=1
            #Loop through all the found connected components
            while i < len(stats):

                # Width of i-th connected component
                curr_connected_comp = stats[i, cv2.CC_STAT_WIDTH]

                # !!! - Determine if the WIDTH is much bigger than the width of a white key and less than 3x? (So we don't get extraneous video text, etc.)
                if curr_connected_comp > self.white_width*1.25 and curr_connected_comp < self.white_width*3:

                    #Threshold just the large component of interest
                    componentMask = (labels == i).astype("uint8") * 255
                    threshMask = cv2.bitwise_and(crop_frame_gray, crop_frame_gray, mask = componentMask) #Replace this with video frame

                    # Histogram segregation of black/white key
                    # Grayscale has one channel so we use [0]
                        #Possible values range from 0 to 256
                    bin_scaler = 4
                    hist = cv2.calcHist([threshMask], [0], None, [256/bin_scaler], [1, 256])

                    #Use a Histogram to compute the dominant non-black (i.e. not the background) colour. Use ~90% of this to threshold the image.
                    T = hist.argmax() * bin_scaler * .95
                    white_notes = threshMask.copy()
                    white_notes[white_notes>T] = 255
                    white_notes[white_notes<T] = 0

                    #Detect the first set of WHITE keys
                    [num_labels_wh, labels_wh, stats_wh, centroids_wh] = connectedComponents(white_notes)

                    #Loop through components and determine which ones may be keys
                    for j in range(1, num_labels_wh):
                        area = stats_wh[j, cv2.CC_STAT_AREA]
                        # !!! - min white pixel area
                        if (20 < area < np.inf): #filtering out relavent detections (the ones big enough to be keys)

                            if j > 1:
                                ## We've added another label
                                num_labels +=1 
                                i +=1

                            #Within labels, we have a matrix that is the same size of the image that holds our split component
                            #First, cut out the original "fat" label - i.e. two or more keys are coupled together
                            coupled_keys_mask = labels != i
                            labels = labels * coupled_keys_mask

                            #Next, increment each label above the cut one up to accomodate the new label
                            higher_mask = labels > i
                            labels = labels + higher_mask

                            #Then append our segregated key
                            new_mask = labels_wh == j
                            new_labels = labels_wh * new_mask
                            new_labels = i * new_labels
                            labels = labels + new_labels

                            ##Remove the original index for the stats and then add the new one
                            if i < len(stats):
                                stats = np.delete(stats,i,0)
                                stats = np.insert(stats,i,stats_wh[j],0)
                            elif j == 1:
                                stats = stats[:-1,:]
                                stats = np.concatenate((stats,stats_wh[j][None,:]),0)
                            else:
                                stats = np.concatenate((stats,stats_wh[j][None,:]),0)

                            ##Remove the original index for the centroids and then add the new one
                            if i < len(centroids):
                                centroids = np.delete(centroids,i,0)
                                centroids = np.insert(centroids,i,centroids_wh[j],0)
                            elif j==1:
                                centroids = centroids[:-1,:]
                                centroids = np.concatenate((centroids,centroids_wh[j][None,:]),0)  
                            else:
                                centroids = np.concatenate((centroids,centroids_wh[j][None,:]),0)  


                            # Plot immediately so indexing doesn't get messed up
                            # !!! - +20 pixels
                            [x,y,w,h,area] = getConnectedComponentRectangle(stats[i,:])
                            y += offset_y_top # We cropped out the first 20 pixels

                            # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                            if w >= self.black_width * 0.5:
                                # For each component, time it and draw via cv2
                                keys_timed_update = timeNotes(x,y,w,h,centroids[i], offset_y_top, h, self.y_coords[1]-offset_y_timing, self.keyboard_array, 
                                                              keys_timed_update, elapsed, output_img,
                                                              font = font, font_scale = 0.5, font_color = (128,0,128))

                        #Detect the next set of keys
                        black_tmp = threshMask.copy()
                        black_tmp[black_tmp>T] = 0 # Invert the coupled image used for the white keys to ID black keys        
                        blurred_black_notes = gaussianBlurring(black_tmp, blur_sq = 5) # Need to blur to remove extraneous detail
                        

                        #Using standard threshold segment just the black keys
                        black_notes = threshold(blurred_black_notes, thresh_type = cv2.THRESH_BINARY)

                        #Detect the second set of keys
                        [num_labels_bl, labels_bl, stats_bl, centroids_bl] = connectedComponents(black_notes)


                        #Loop through components and determine which ones may be keys
                        for k in range(1, num_labels_bl):
                            area = stats_bl[k, cv2.CC_STAT_AREA]

                            # !!! - min area
                            if (20 < area < np.inf): #filtering out relavent detections (the ones big enough to be keys)

                                if k > 1:
                                    ## We've added another label
                                    num_labels +=1 
                                    i+=1

                                #For the second set of keys WE DON'T NEED TO CUT anything
                                # coupled_keys_mask = labels != i
                                # labels = labels * coupled_keys_mask

                                #Next, increment each label above the cut one up to accomodate the new label
                                higher_mask = labels > i + 1
                                labels = labels + higher_mask

                                #Then append our segregated key
                                new_mask = labels_bl == k
                                new_labels = labels_bl * new_mask
                                new_labels = (i + 1) * new_labels
                                labels = labels + new_labels

                                ##Add
                                if i < len(stats):
                                    stats = np.insert(stats,(i+1),stats_bl[k],0)                       
                                else:
                                    stats = np.concatenate((stats,stats_bl[k][None,:]),0)


                                ##Add
                                if i < len(centroids):
                                    centroids = np.insert(centroids,(i+1),centroids_bl[k],0)
                                else:
                                    centroids = np.concatenate((centroids,centroids_bl[k][None,:]),0)

                                #Plot immediately so indexing doesn't get messed up
                                # !!! - +20 pixels
                                [x,y,w,h,area] = getConnectedComponentRectangle(stats[i+1,:])
                                y += offset_y_top # We cropped out the first 20 pixels


                                # !!! - minimum black width
                                # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                                if w >= self.black_width * 0.5:
                                    # For each component, time it and draw via cv2
                                    keys_timed_update = timeNotes(x,y,w,h,centroids[i+1], offset_y_top, h, self.y_coords[1]-offset_y_timing, self.keyboard_array, 
                                                              keys_timed_update, elapsed, output_img,
                                                              font = font, font_scale = 0.5, font_color = (0,0,255))

                else:
                    # !!! - +20 pixels
                    [x,y,w,h,area] = getConnectedComponentRectangle(stats[i,:])
                    y += offset_y_top # We cropped out the first 20 pixels

                    # Ensure our detected note is bigger than the median width of a black key multiplied by a factor
                    if (20 < area < np.inf) and w >= self.black_width * 0.5: #filtering out relavent detections (the ones big enough to be keys)
                        # For each component, time it and draw via cv2
                        keys_timed_update = timeNotes(x,y,w,h,centroids[i], offset_y_top, h, self.y_coords[1]-offset_y_timing, self.keyboard_array, 
                                                              keys_timed_update, elapsed, output_img,
                                                              font = font, font_scale = 0.33, font_color = (255,255,255))

                i+=1


            #Show the frame + drawn rectangle
            cv2.imshow("Video", output_img)

            #Can break early by pressing "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) & 0xFF == ord("p"):
                cv2.waitKey(-1) #wait until any key is pressed

        # print(keys_timed_update)    
        camera.release()
        cv2.destroyAllWindows()
        self.timed_keys = keys_timed_update

    ################################################################################

    def export_notes(self):
        new_list = []

        for i in range (len(self.timed_keys)):
            temp = []
            temp.append(self.timed_keys[i][0])
            if (len(self.timed_keys[i]) > 1):
                for j in range(1,len(self.timed_keys[i])):
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

if __name__ == "__main__":
    #Download a YouTube video and process it to determine where the centroid of the black and white keys are
    
    keyboard = Keyboard('https://www.youtube.com/watch?v=sleZ-hzrtRY&ab_channel=Marioverehrer')
    keyboard.downloadYouTube()
    keyboard.speed_video(spd_factor = 5)
    keyboard.detect_keys(resize_width = 600, bl_blur_sq = 5, bl_canny_th1 = 200, bl_canny_th2 = 200, bl_thresh1 = 90, bl_thresh2 = 150, wh_blur_sq = 7)
    keyboard.video_process()
    keyboard.export_notes()

# [black, white, black_width, white_width] = keyboard.getKeys()     # Array of keys
# key_img = keyboard.getFrame()                                     # Image of video used to detect our 88 keys
# keyboard_array = keyboard.getFullKeyList()                        # Combined array of keys
# keyboard_y_coords = keyboard.getKeyboardYCoords()                 # y[0] is the bottom of the keyboard, y[1] is the top
# timed_keys = keyboard.getTimedKeys()                                # Array of timed keys