from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# just  a function for printing images
def display_img(title, img):
    cv2.imshow(title, img)


def getNextNote(first_note):
    if first_note == "A#":
        return "C#"
    if first_note == "C#":
        return "D#"
    if first_note == "D#":
        return "F#"
    if first_note == "F#":
        return "G#"
    if first_note == "G#":
        return "A#"


#sample piano image from youtube
img = cv2.imread("piano_img.png")
#converting to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edge detection 
edges = cv2.Canny(gray, 100,100, apertureSize=3)

##########################################
### Using hough transform to identify lines 
lines = cv2.HoughLines(edges, 1, np.pi/180, 400)
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
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255),2)
    display_img("dslkd", img)
    cv2.waitKey(0)

#sorting list and getting rid of smallest value
y_cord.sort(reverse=True)
y_cord.pop()

#cropping the image based on y_cord list
crop_img = img[int(y_cord[1])+20:int(y_cord[0])]
crop_blur = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#display_img("crop_blur", crop_blur)

#Using standard threshold to create contrast between white/black keys
_, th1 = cv2.threshold(crop_blur, 85, 150, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
display_img("threshold", th1)


####################################
####### using connected component detection algorithm to seperate all the black notes
connectivity = 8
output = cv2.connectedComponentsWithStats(th1, connectivity, cv2.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]
print(centroids.shape)

final_labels = []

#For loop only used for displaying 
for i in range(1, num_labels):
    if i == 0:
        text = "examining component {}/{} (background)".format(i + 1, num_labels)
    else:
        text = "examining component {}/{}".format(i+1, num_labels)

    print("[INFO] {}".format(text))
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    output = th1.copy()
    if (1000 < area < 2000): #filtering out relavent detections (the ones big enough to be black keys)
        final_labels.append(i)
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,255,0),3)
        cv2.circle(output, (int(cX), int(cY)), 4, (255,255,0), -1)
        componentMask = (labels == i).astype("uint8") * 255
        display_img("Output", output)
        display_img("Connected Component", componentMask)
        cv2.waitKey(0)

#just for visualization lol
for i in range(len(final_labels)):
    xc,yc = centroids[final_labels[i]]
    #x1 = stats[final_labels[i], cv2.CC_STAT_LEFT]
    #del_x = stats[final_labels[i], cv2.CC_STAT_WIDTH]
    #lol = cv2.line(img,(int(x1),0),(int(x1),900),(0,255,0),1)
    #lol = cv2.line(img,(int(x1+del_x),0),(int(x1+del_x),900),(0,255,0),1)
    lol = cv2.line(img,(int(xc),0),(int(xc),900),(0,0,255),1)
    cv2.imshow("lol", lol)
    cv2.waitKey(0)

#Printing out the difference between black keys
#figure out someway to normalize this data so 
#it works with all figures
difference = []
for i in range(len(final_labels)-1):
    x2,y2 = centroids[final_labels[i+1]]
    x1,y1 = centroids[final_labels[i]]
    diff = x2-x1
    difference.append(diff)
    print(diff)

#plotting distance between black keys vs centroids of all black keys
x_axis = []
for i in range(len(final_labels)-1):
    x = centroids[final_labels[i]][0]
    #print(x)
    x_axis.append(x)

plt.plot(x_axis, difference)
plt.show()

#checking the difference for the next three notes to figure out
#which note is being played
##########
#! JUST A PROTOTYPE - NEEDS TO BE WOKED ON
#right now it's hardcoded with numbers (40,20).. but needs to be normalized
#########
first_note = None
if (difference[0] > 40):
    if (difference[1] > 40):
        pass
    else:
        if (difference[2] > 40):
            first_note = 'A#'
            print(first_note)

black_key_dict = {}
for i in range(len(final_labels)):
    x = centroids[final_labels[i]][0]
    black_key_dict[x] = first_note
    first_note = getNextNote(first_note)

print(black_key_dict)


#Labeling all the black keys
for centroid in black_key_dict:
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 350
    lol = cv2.line(img,(int(centroid),0),(int(centroid),900),(0,0,255),1)
    lol = cv2.putText(img, black_key_dict[centroid], (int(centroid), y), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
    cv2.imshow("lol", lol)
    cv2.waitKey(0)

##white key labelling
#! NEED TO IMPLEMENT SPECIAL CONDITION FOR LAST WHITE KEYS 
white_key_dict = {}
for i in range(len(final_labels)-1):
    x = centroids[final_labels[i]][0]

    if (difference[i] > 40):
        origin = x + difference[i]/2
        delta_x = difference[i]/4
        lol = cv2.line(img,(int(origin+delta_x),0), (int(origin+delta_x),900), (0,255,0),1)
        lol = cv2.line(img,(int(origin-delta_x),0), (int(origin-delta_x),900), (0,255,0),1)
        cv2.imshow("lol", lol)
        cv2.waitKey(0)
    else:
        delta_x = difference[i]/2
        lol = cv2.line(img,(int(x+delta_x),0), (int(x+delta_x),900), (0,255,0),1)
        cv2.imshow("lol", lol)
        cv2.waitKey(0)

#Using standard threshold to create contrast between white/black keys
kernel = np.ones((20,1), np.uint8)
_, th2 = cv2.threshold(crop_blur, 85, 150, cv2.THRESH_BINARY_INV)
d_im = cv2.dilate(th2, kernel, iterations=2)
e_im = cv2.erode(d_im, kernel, iterations=2)
display_img("threshold", d_im)
display_img("threshold", e_im)


#cv2.imshow('cropped',crop_img)
#display_img("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



