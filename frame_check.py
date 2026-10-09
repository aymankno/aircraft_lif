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

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print()
    while True:
        minute_start = int(input("Minutes (start): "))
        if minute_start == "q":
            break
        second_start = int(input("Seconds (start): "))
        frame_start = int(((minute_start * 60) + (second_start)) * fps)
        print(f"Start: {frame_start}")

        minute_end = int(input("Minutes (end): "))
        second_end = int(input("Seconds (end): "))
        frame_end = int(((minute_end * 60) + (second_end)) * fps)
        print(f"End: {frame_end}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        current_frame = frame_start

        while True:
            ret, frame = cap.read()
            cv2.imshow("VIDEO9", frame)
            key = cv2.waitKey(33) & 0xFF
            current_frame += 1
            if current_frame == frame_end:
                cap.destroyAllWindows()
                break
            elif key == ord("s"):
                cap.destroyAllWindows()
                break

        event = input("Event was: t/g (g), takeoff (t), or landing (l)? ")
        if event == "t":
            event = "takeoff"
        if event == "l":
            event = "landing"
        if event == "g":
            event = "touch/go"
        if event == "a":
            event = "go-around"

        dist = input("Far (f), near (n), or both (b? ")
        if dist == "f":
            dist = "far"
        if dist == "n":
            dist = "near"
        if dist == "b":
            dist = "both"
            
        print(f"{event.upper()}: {frame_start}:{frame_end}, {dist}")
        save = input("Confirm (y) ")
        if save == "y":
            writer.writerow([frame_start, frame_end, dist, event])
            print("Saved")
        else:
            print("Event not saved. Canceled")

        










