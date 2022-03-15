# piano2sheet

# Introduction
In this project, the objective is to convert [Synthesia](https://en.wikipedia.org/wiki/Synthesia) videos and videos of piano players that utilize MIDI visualizations to sheet music. 

## Video Analysis 
We created a pipeline that allows identification of black and white keys seperately and labels each note.

**Image of Raw Video:**

<img src="/README_images/piano_roll.png" alt="Raw video" width="720"/>

**Image after image computer vision pipeline:**

<img src="/README_images/piano_roll_ident.png" alt="Raw video" width="720"/>

**MIDI:**

<img src="/README_images/MIDI.png" alt="Raw video" width="720"/>

## Next Steps
To create web app (android and ios app pending)

## Technical Details

As MIDI keyboards are capable of [exporting key presses to Synthesia](https://www.synthesiagame.com/keyboards/Help/desktop/midi) in a very systematic manner, we can use this to detect which keys are being played, and when.

>**1. The video output is consistently ordered**

There are 52 white keys and 36 black keys on a standard piano.
<img src="/README_images/1_piano.png" alt="Raw video" width="600"/>


>**2. The key portion can be extracted**

We can crop and threshold the image to identify the keys.

<img src="/README_images/2_piano_thresholded.png" alt="Extracted keys" width="600"/>

>**3. We can run detection algorithms on the thresholded image**

Check out this GIF that highlights the detected black keys. This can be similarly performed for the white keys.

<img src="/README_images/3_detect_black_keys.gif" alt="Detected keys" width="600"/>

>**4. From any image, we can perform the same thresholding**

Side-by-side comparison of the raw image and thesholded image.

<p float="left">
  <img align='top' src="/README_images/4_raw_snapshot.png" alt="Detected keys" width="400"/>
  <img align='top' src="/README_images/5_thresholded_snapshot.png" alt="Thresholded keys" width="400"/>
</p>

>**5. Fully connected notes need to be segmented**

The right image shows two distinctly coloured notes that can be thresholded by using a histogram and identifying the peak.

<p float="left">
  <img align='top' src="/README_images/4_raw_snapshot.png" alt="Detected keys" width="400"/>
  <img align='top' src="/README_images/6_segmented_snapshot.png" alt="Connected keys" width="400"/>
</p>

The result is:
<p float="left">
  <img align='top' src="/README_images/7_pinpoint.png" alt="Segmented keys (left)" width="400"/>
  <img align='top' src="/README_images/9_pinpoint2.png" alt="Segmented keys (right)" width="400"/>
</p>

>**6. And the full detection run on the image**

<img src="/README_images/10_output.png" alt="Full image detection" width="600"/>


>**7. Perform this on every frame of the video**

Below is a GIF showing a short-run of the detection algorithm on the keys!

<img src="/README_images/11_snippet.gif" alt="Video detection" width="600"/>

