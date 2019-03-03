
# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)'''
    # Find and sort the contours of all cards in the image (query cards)
    x=0
    y=0
    w=200
    h=450
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    pre_proc_card = Cards.preprocess_corner(x, y, w, h, image)

                # Find the best rank and suit match for the card.
    best_rank_match,best_suit_match,rank_diff,suit_diff = Cards.match_card(pre_proc_card,train_ranks,train_suits)
    print(best_rank_match)


    # Draw card contours on image (have to do contours all at once or
    # they do not show up properly for some reason)
    '''if (len(cards) != 0):
        temp_cnts = []
        for i in range(len(cards)):
            temp_cnts.append(cards[i].contour)
        cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
       ''' 
        
    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector",image)

    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()

