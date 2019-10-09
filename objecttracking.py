import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
kernel = np.ones((3, 3), np.uint8)
kernel_2 = np.ones((5, 5), np.uint8)

# dis_color_1 = np.zeros((300, 500, 3), np.uint8)
# dis_color_2 = np.zeros((300, 500, 3), np.uint8)
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:

    # Take each frame
    _, frame = cap.read()
    # frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)


    # define range of orange color in HSV
    # lower_orange = np.array([0, 100, 100])
    # upper_orange = np.array([20, 255, 255])

    # define range of color in HSV
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_p = np.array([u_h, u_s, u_v])

    # dis_color_1[:] = l_b
    # dis_color_2[:] = u_p
    #
    # cv2.imshow('Tracking', dis_color_1)
    # cv2.imshow('Tracking', dis_color_2)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, l_b, u_p)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_2)

    # _, mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    planets = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=80,
                               param2=25, minRadius=30, maxRadius=0)

    print(f'circles = {circles}')

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw	the	outer	circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 10)
            # draw	the	center	of	the	circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, hsv, mask=mask)
    # res = cv2.bitwise_and(hsv, frame, mask=mask)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
