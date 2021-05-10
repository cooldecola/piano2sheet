from cv2 import cv2
import numpy as np



def display_img(title, img):
    cv2.imshow(title, img)



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
####### using connected component detection algorithm to seperate each individual notes
connectivity = 8
output = cv2.connectedComponentsWithStats(th1, connectivity, cv2.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

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
    if (1000 < area < 2000):
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,255,0),3)
        cv2.circle(output, (int(cX), int(cY)), 4, (255,255,0), -1)
        componentMask = (labels == i).astype("uint8") * 255
        display_img("Output", output)
        display_img("Connected Component", componentMask)
        cv2.waitKey(0)


#cv2.imshow('cropped',crop_img)
#display_img("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
