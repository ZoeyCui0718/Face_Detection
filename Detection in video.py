import cv2

def facedetect(image):
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # To use OpenCV recognition classifier
    classifier = cv2.CascadeClassifier(
        "C:/Python38/Lib/site-packages/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
    )

    # To define the color of rectangles/circles
    color = (0, 255, 0)

    # Detect faces
    faceRects = classifier.detectMultiScale(
        grayimage, scaleFactor=1.2, minNeighbors=3, minSize=(32,32)
    )

    # if detecting faces
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            # frame the face
            cv2.rectangle(image, (x,y), (x+h, y+w), color, 2)

    cv2.imshow("image", image)

# To get the first camera
capture = cv2.VideoCapture(0)

# show frame by frame
while True:
    ret, frame = capture.read()
    facedetect(frame)
    # To get the ASCII code of the last character the user entered
    # If the user entered "q", then break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the camera
capture.release()
cv2.destroyAllWindows()
