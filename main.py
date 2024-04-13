import cv2
from cvzone import HandTrackingModule, overlayPNG
import numpy as np
from supportFunction import *

# Read the intro, kill, and winner images from files
intro = cv2.imread("frames/img1.jpeg")
kill = cv2.imread("frames/img2.png")
winner = cv2.imread("frames/img3.png")

# Read the intro, kill, and winner images from files
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the HandDetector object from the HandTrackingModule
detector = HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.77)

# Initialize the HandDetector object from the HandTrackingModule
sqr_img = cv2.imread("img/sqr(2).png")
foreground = resize_image_with_padding(sqr_img, 1280, 720)

# Initialize the HandDetector object from the HandTrackingModule
gameOver = False
NotWon = True
gameON = False
startingCondition = 0
error = 0

# cv2.imshow('Squid Game', kill)
# cv2.waitKey()

# Initialize the HandDetector object from the HandTrackingModule
pixels_in_rectangle = get_pixels_in_rectangle(foreground)
# for pixel in pixels_in_rectangle:   # Hiển thị vùng cắt bánh đúng
#     foreground[pixel[0], pixel[1]] = [0, 255, 0]
# cv2.waitKey()
calcPixel = 0
while True:  # Main loop for the game
    while True:  # Display the intro screen until 'q' is pressed
        cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=1.3, fy=1.3))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    gameOver = False  # Main loop for the game
    startingCondition = 0
    error = 0
    foreground = resize_image_with_padding(sqr_img, 1280, 720)
    while not gameOver:  # Main loop for the game
        alpha = 0.3  # Main loop for the game
        _, frame = cam.read()  # Main loop for the game
        background = cv2.flip(frame, 1)
        hands, img = detector.findHands(
            background, flipType=False)  # Detect hands in the frame
        added_image = cv2.addWeighted(
            background, alpha, foreground, 1 - alpha, 0)  # Blend foreground and background images

        if hands:  # Process hand gestures if hands are detected
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            finger = lmList1[8][:2]  # Get the position of the index finger
            # Check if game is in progress ( finger is within the rectangle region)
            if startingCondition >= 50:
                gameON = True
                cv2.putText(foreground, 'Game On......', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            if finger[::-1] in pixels_in_rectangle:
                added_image = cv2.circle(
                    added_image, finger, 5, color_true, -1)
                startingCondition += 1
                if gameON:
                    cv2.circle(foreground, finger, 3, (0, 0, 255), -1)
            else:
                added_image = cv2.circle(
                    added_image, finger, 7, color_false, -1)
                if gameON:
                    error += 1
                    if error >= 5:
                        gameON = False
                        startingCondition = 0
                        error = 0
                        gameOver = True
                else:
                    startingCondition = 0

        cv2.imshow('Squid Game', added_image)  # Display the game screen

        if (cv2.waitKey(20) & 0xFF == ord('q')):
            gameOver = True
            break
    while True:  # Display the kill screen until 'q' is pressed
        # resized_image = cv2.resize(added_image, (300, 300))
        resized_image = cv2.resize(added_image, (300, 300), fx=0.69, fy=0.69)
        top_right_corner = (0, kill.shape[1] - resized_image.shape[1])
        kill[top_right_corner[0]:resized_image.shape[0],
             top_right_corner[1]:kill.shape[1]] = resized_image
        cv2.imshow('Squid Game', cv2.resize(kill, (0, 0), fx=0.69, fy=0.69))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
