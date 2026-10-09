import cv2
import pandas as pd
import csv

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
video10 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/10muted.mp4"

cap = cv2.VideoCapture(video9)
backSub = cv2.createBackgroundSubtractorMOG2()
key = None
current_frame = 0

with open('video9_labels.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "area", "movement", "type"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if key == ord('s'):
            start = current_frame
            print("start")

        elif key == ord('t'):
            if start != None:
                area = "far"
                end = current_frame
                writer.writerow([start, end, "takeoff"])
                f.flush()
                print(f"TAKEOFF:, {start}, {end}, {area}")
                start = None
                area = None
        elif key == ord('l'):
            if start != None:
                area = "near"
                end = current_frame
                writer.writerow([start, end, "landing"])
                f.flush()
                print(f"LANDING:, {start}, {end}, {area}")
                start = None
                area = None
        elif key == ord("g"):
            if start != None:
                area = "both"
                end = current_frame
                writer.writerow([start, end, area, "touch_go"])
                f.flush()
                print(f"TOUCH/GO:, {start}, {end}, {area}")
                start = None
                area = None

        # channel 2; two planes in one shot
        elif key == ord("1"):
            start_1 = current_frame
            print("start on second channel. t/o: 2, landing: 3, t/g: 4")
        elif key == ord("2"):
            if start_1 != None:
                area_1 = "far"
                end_1 = current_frame
                writer.writerow([start, end, "takeoff"])
                f.flush()
                print(f"TAKEOFF chnl 2:, {start}, {end}, {area}")
                start = None
                area = None
        elif key == ord('3'):
            if start_1 != None:
                area_1 = "near"
                end_1 = current_frame
                writer.writerow([start_1, end_1, "landing"])
                f.flush()
                print(f"LANDING chnl 2:, {start_1}, {end_1}, {area_1}")
                star_1 = None
                area_1 = None
        elif key == ord("4"):
            if start_1 != None:
                area_1 = "both"
                end_1 = current_frame
                writer.writerow([start_1, end_1, area_1, "touch_go"])
                f.flush()
                print(f"TOUCH/GO chnl 2:, {start_1}, {end_1}, {area_1}")
                start_1 = None
                area_1 = None


        elif key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break


        cv2.imshow("FRAME CHECK - VIDEO 9", frame)
        key = cv2.waitKey(33) & 0xFF
        current_frame += 1