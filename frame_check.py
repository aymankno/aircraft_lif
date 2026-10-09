import cv2
import pandas as pd
import csv

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
video10 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/10muted.mp4"

cap = cv2.VideoCapture(video9)
backSub = cv2.createBackgroundSubtractorMOG2()
end = 0
key = None

with open('video9_labels.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "area", "movement", "type"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if key == ord('1'):
            start = current_frame
            print("Start - close")
        if key == ord('2'):
            start_2 = current_frame
            print("Start - far")

        elif key == ord('t'):
            if start != None:
                area = input("far (f) or near (n)? ")
                if area == "f":
                    area = "far"
                if area == "n":
                    area = "near"
                else:
                    area = ("Invalid. Type 'far' or 'near' ")
                end = current_frame
                writer.writerow([start, end, "takeoff"])
                f.flush()
                print(f"TAKEOFF:, {start}, {end}, {area}")
                start = None
                area = None
        elif key == ord('l'):
            if start != None:
                area = input("far (f) or near (n)? ")
                if area == "f":
                    area = "far"
                if area == "n":
                    area = "near"
                else:
                    area = ("Invalid. Type 'far' or 'near' ")
                end = current_frame
                writer.writerow([start, end, "landing"])
                f.flush()
                print(f"LANDING:, {start}, {end}, {area}")
                start = None
                area = None
        elif key == ord("g"):
            if start != None:
                area = input("far (f) or near (n)? ")
                if area == "f":
                    area = "far"
                if area == "n":
                    area = "near"
                else:
                    area = ("Invalid. Type 'far' or 'near' ")
                end = current_frame
                writer.writerow([start, end, area, "touch_go"])
                f.flush()
                print(f"TOUCH/GO:, {start}, {end}, {area}")
                start = None
                area = None

        elif key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break


        cv2.imshow("FRAME CHECK - VIDEO 9", frame)
        key = cv2.waitKey(33) & 0xFF
        current_frame += 1