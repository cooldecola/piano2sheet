import cv2 as cv2

# Function for Gaussian Blurring
def gaussianBlurring(gray_img, blur_sq, std_dev = 0):
    return cv2.GaussianBlur(gray_img, (blur_sq, blur_sq), std_dev)

# Function for Canny Edge Detection
def cannyDetection(blurred_img, th1, th2, apertureSize = 3):
    return cv2.Canny(blurred_img, th1, th2, apertureSize)

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
