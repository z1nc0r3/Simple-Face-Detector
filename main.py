import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

while (True) :
    ret, frame = video.read()
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_positions = trained_face_data.detectMultiScale(gray_img)
    
    if len(face_positions) > 0 :
        for i in face_positions:
            (x, y, w, h) = i
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('live', frame)
    
    key = cv2.waitKey(1)
    
    if key == 32 :
        break;
