from cv2 import cv2
import statistics
import numpy as np
import bisect        # Key insert point

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
    #print(y_cord[0:2])
    return y_cord[0:2]


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
                #componentMask = (labels == i).astype("uint8") * 255
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
