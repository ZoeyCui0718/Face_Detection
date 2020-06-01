import cv2
from keras.models import load_model
import numpy as np
import datetime

startTime = datetime.datetime.now()
emotion_classifier = load_model('classifier/emotion_models/simple_CNN.530-0.65.hdf5')
endTime = datetime.datetime.now()
print(endTime-startTime)

emotion_labels = {0: 'angry', 1: 'hate', 2: 'afraid', 3:'happy', 4: 'sad', 5: 'surprised', 6: 'peaceful'}

filepath = "C:/Users/Zoey.Cui/Desktop/face1.jpg"

# Read the image
image = cv2.imread(filepath)
# Convert to gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Call OpenCV facial recognition classifier
classifier = cv2.CascadeClassifier(
    "C:/Python38/Lib/site-packages/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
)

faces = classifier.detectMultiScale(
    gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 0, 0)

for (x, y, w, h) in faces:
    gray_face = gray_image[y:(y + h), x:(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.rectangle(image, (x + 10, y + 10), (x + h - 10, y + w - 10),
                  (255, 255, 255), 2)
    image = cv2.putText(image, emotion, x + h * 0.3, y, color, 20)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

