import cv2

filepath = "C:/Users/Zoey.Cui/Desktop/face2.jpg"

# Read the image
image = cv2.imread(filepath)
# Convert to gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use OpenCV facial recognition classifier
classifier = cv2.CascadeClassifier(
    "C:/Python38/Lib/site-packages/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
)

# Define the color to draw rectangle/circles
color = (0, 255, 0)

# To detect faces
faceRects = classifier.detectMultiScale(
    gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0:  # If detecting faces
    for faceRect in faceRects:
        x, y, w, h = faceRect
        # face
        cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
        # left eye
        cv2.circle(image, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        # right eye
        cv2.circle(image, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        # mouth
        cv2.rectangle(image, (x + 3 * w // 8, y + 3 * h // 4),
                      (x + 5 * w // 8, y + 7 * h // 8), color)

# Show images
cv2.imshow("image", image)
c = cv2.waitKey(10)
cv2.waitKey(0)
cv2.destroyAllWindows()
