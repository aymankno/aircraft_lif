import cv2

cap = cv2.VideoCapture("/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30800)
ret, frame = cap.read()

boxes = cv2.selectROIs("pick regions", frame)   # array of (x, y, w, h)
runway_box, helipad_box = boxes[0], boxes[1]

print(boxes.tolist())