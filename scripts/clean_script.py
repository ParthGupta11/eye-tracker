"""
Program to take video feed from webcam and apply eye tracking
"""

import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, shape, side="right", verbose=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pupil_found = False
    is_straight = False
    try:
        socket_inner = (0, 0)
        socket_outer = (0, 0)
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        pupil_found = True

        if side == "right":
            cx += mid
            socket_inner = shape[42]
            socket_outer = shape[45]
        else:
            socket_inner = shape[39]
            socket_outer = shape[36]

        
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        cv2.circle(img, socket_inner, 4, (0, 255, 0), 2)    
        cv2.circle(img, socket_outer, 4, (0, 255, 0), 2)

        socket_width = abs(socket_inner[0] - socket_outer[0])    
        inner2pupil = abs(socket_inner[0] - cx)
        outer2pupil = abs(socket_outer[0] - cx)

        if inner2pupil < socket_width/4 or outer2pupil < socket_width/4:
            if verbose:
                print("\033[1;31m [{}] \t 2inner, 2outer: {}, {} \t LOOKING AWAY!".format(side, inner2pupil, outer2pupil))
            is_straight = False
        else:
            if verbose:
                print("\033[1;32m [{}] \t 2inner, 2outer: {}, {} \t Looking straight".format(side, inner2pupil, outer2pupil))
            is_straight = True  
        return is_straight
    except:
        # print("exception occured")
        pass
    finally:
        return pupil_found, is_straight

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models\\shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while(True):
    # reading image from webcam
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting to grayscale
    rects = detector(gray, 1)
    for rect in rects:

        # predicting facial keypoints
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # preparing segmenatation mask for eyes using keypoints
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)

        # segmenting out eye using mask
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        # thresholding based segmenatation for pupils
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)

        # processing pupil segmentation mask
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.dilate(thresh, None, iterations=2) 
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.medianBlur(thresh, 3)

        # tracking the pupils using contouring
        pupil_found_left, is_straight_left = contouring(thresh[:, 0:mid], mid, img, shape, side="left", verbose=False)
        pupil_found_right, is_straight_right = contouring(thresh[:, mid:], mid, img, shape, side="right", verbose=False)

        # generating message based on contouring
        if pupil_found_left or pupil_found_right:
            print("\033[1;37mLeft: {}, Right: {}".format(is_straight_left, is_straight_right))

            if (pupil_found_left and not is_straight_left) or (pupil_found_right and not is_straight_right):
                print("\033[1;31mLOOK INTO THE SCREEN PLEASE!")
            else:
                print("\033[1;32mLooking straight")
        else:
            print("\033[1;37mCalibration Required.")

        # show the image with the face detections + facial landmarks
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

    # display windows
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)

    # breakpoint to exit the program
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# releasing resources 
cap.release()
cv2.destroyAllWindows()