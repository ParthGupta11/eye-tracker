"""
Program to detect face and facial keypoints of given images using the dlib library
"""

import cv2
import numpy as np
import dlib

left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
img_path = "test_images\\face1.png"
model_path = "models\\shape_68.dat"

def rect_to_bb(rect):
    """
    Take a dlib bouding box and convert it to (x, y, w, h) format used in opencv
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    """ 
    Converts 68 facial keypoints shape to numpy array
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_on_mask(shape, mask, side):
    """
    Convex filling to create mask for eye keypoints
    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1) # rects contains all the faces detected

    bboxes = []
    for rect in rects:
        bbox = rect_to_bb(rect)
        bboxes.append(bbox)
        x, y, w, h = bbox
        start_pt = (x, y)
        end_pt = (x+w, y+h)
        cv2.rectangle(img, start_pt, end_pt, (255, 0, 0), 2)
        cv2.imshow("Bouding Boxes", img)
    
    predictor = dlib.shape_predictor(model_path)
    for (i, rect) in enumerate(rects):
        # predicting facial keypoints
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # # plotting keypoints
        # for (x, y) in shape:
        #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # creating mask for eyes
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # cv2.imshow("Blank mask", mask)
        mask = eye_on_mask(shape, mask, left)
        # cv2.imshow("Left eye mask", mask)
        mask = eye_on_mask(shape, mask, right)
        # cv2.imshow("Left + Right eye mask", mask)

        # processing the mask
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("Segmented eyes", eyes)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        cv2.imshow("eye mask", eyes_gray)

        # thresholding to segement eyeballs
        def nothing(x):
            pass
        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)

    cv2.imshow("img", img)

# cap = cv2.VideoCapture(0)
# while(True):
#     ret, img = cap.read()
#     detect_face(img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
#         break

img = cv2.imread(img_path)
detect_face(img)
cv2.waitKey(0)
cv2.destroyAllWindows()