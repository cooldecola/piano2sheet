import cv2

def get_black_key(image, action):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if action == 'NO_FILTER':
        filtered = img
    elif action == 'BLACK_KEYS':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif action == 'WHITE_KEYS':
        filtered = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGBA)

    return filtered
