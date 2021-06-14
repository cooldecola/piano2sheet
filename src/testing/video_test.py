from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# just  a function for printing images
def display_img(title, img):
    cv2.imshow(title, img)

#Only considering sharps for now
def getNextSharp(first_note):
    if first_note == "A#":
        return "C#"
    elif first_note == "C#":
        return "D#"
    elif first_note == "D#":
        return "F#"
    elif first_note == "F#":
        return "G#"
    elif first_note == "G#":
        return "A#"
    

#Flats
def getNextFlat(first_note):
    if first_note == "B♭":
        return "D♭"
    elif first_note == "D♭":
        return "E♭"
    elif first_note == "E♭":
        return "G♭"
    elif first_note == "G♭":
        return "A♭"
    elif first_note == "A♭":
        return "B♭"
      
#White notes
def getNextNote(first_note):
    if first_note == "A":
        return "B"
    elif first_note == "B":
        return "C"
    elif first_note == "C":
        return "D"
    elif first_note == "D":
        return "E"
    elif first_note == "E":
        return "F"
    elif first_note == "F":
        return "G"
    elif first_note == "G":
        return "A"



cap = cv2.VideoCapture("88_key_video.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    
    #converting to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #edge detection 
    std_dev = 0
    k = 5
    t1 = 75
    t2 = 100
    blurred = cv2.GaussianBlur(gray, (k,k), std_dev)
    edges = cv2.Canny(blurred, t1,t2, apertureSize = 3)

    ##########################################
    ### Using hough transform to identify lines 
    lines = cv2.HoughLines(edges, 1, np.pi/180, 300)
    #empty list - will be added with pixels that correspond with 
    #the lines generated by hough transform 
    y_cord = []

    #iterating through lines
    for line in lines: 
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        y_cord.append(y0) #appending to list
        x1 = int(x0 + 1500 * (-b))
        y1 = int(y0 + 1500 * (a))
        x2 = int(x0 - 1500 * (-b))
        y2 = int(y0 - 1500 * (a))
        cv2.line(gray, (x1,y1), (x2,y2), (0,255,0),2)

    if (len(y_cord) > 2):
        y_cord.sort(reverse=True)
        y_cord.pop()

    #cropping the image based on y_cord list
    crop_img = gray[int(y_cord[1])+20:int(y_cord[0])]

    #Using standard threshold to create contrast between white/black keys
    _, th1 = cv2.threshold(crop_img, 85, 150, cv2.THRESH_BINARY_INV)

    #####################################################################################
    ####### using connected component detection algorithm to separate all the black notes
    connectivity = 1
    output = cv2.connectedComponentsWithStats(th1, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    print(num_labels)

    # print(centroids.shape)
    final_labels_bl = []
    output = crop_img.copy()

    #For loop only used for displaying 
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if (100 < area < np.inf): #filtering out relavent detections (the ones big enough to be black keys)
            final_labels_bl.append(i)
            componentMask = (labels == i).astype("uint8") * 255

    print(len(final_labels_bl))

    cv2.imshow('frame', th1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break