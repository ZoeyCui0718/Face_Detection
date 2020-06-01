import cv2
from keras.models import load_model
import numpy as np
import tensorflow

filepath = "C:/Users/Zoey.Cui/Desktop/face1.jpg"

# Read the image
image = cv2.imread(filepath)
# Convert to gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Call OpenCV facial recognition classifier
classifier = cv2.CascadeClassifier(
    "C:/Python38/Lib/site-packages/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
)

faces = classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(140,140))

gender_classifier = load_model("classifier/gender_models/simple_CNN.81-0.96.hdf5")
gender_labels = {0: 'female', 1: 'male'}
color = (255, 255, 255)

for (x, y, w, h) in faces:
    face = image[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
    image = cv2.putText(image, gender, x + h, y, color, 30)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

