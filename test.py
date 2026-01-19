import cv2

cap = cv2.VideoCapture("left.mp4")
ret, frame = cap.read()
print("Readable:", ret)
cap.release()
