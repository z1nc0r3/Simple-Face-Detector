import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('grammar-devotional.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_position = trained_face_data.detectMultiScale(gray_img)

print ("number of peoples =" , len(face_position));

for i in face_position:
    (x, y, w, h) = i
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('simple face detector', img)
cv2.waitKey()
