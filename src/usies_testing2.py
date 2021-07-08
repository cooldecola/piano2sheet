from __future__ import unicode_literals


from keyboard import Keyboard


if __name__ == "__main__":
    #Download a YouTube video and process it to determine where the centroid of the black and white keys are
    keyboard_array = []
    white_width = 0.0
    black_width = 0.0 
    keyboard = Keyboard('https://www.youtube.com/watch?v=sleZ-hzrtRY&ab_channel=Marioverehrer', keyboard_array, white_width, black_width)
    keyboard.downloadYouTube()
    #keyboard.speed_video(spd_factor = 5)
    keyboard.detect_keys(resize_width = 600, bl_blur_sq = 5, bl_canny_th1 = 200, bl_canny_th2 = 200, bl_thresh1 = 90, bl_thresh2 = 150, wh_blur_sq = 7)
    keyboard.video_process()
    keyboard.export_notes()

# [black, white, black_width, white_width] = keyboard.getKeys()     # Array of keys
# key_img = keyboard.getFrame()                                     # Image of video used to detect our 88 keys
# keyboard_array = keyboard.getFullKeyList()                        # Combined array of keys
# keyboard_y_coords = keyboard.getKeyboardYCoords()                 # y[0] is the bottom of the keyboard, y[1] is the top
# timed_keys = keyboard.getTimedKeys()                                # Array of timed keys