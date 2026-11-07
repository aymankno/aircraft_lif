from superneuromat import SNN
import cv2
import numpy as np
import os
import sys
import time

import ac_init

here = os.path.dirname(os.path.abspath(__file__))
trained_weights = os.path.join(here, "trained_weights.npy")
video7 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/7muted.mp4'

net = ac_init.build_network()
net.snn.set_weights_from_mat(np.load(trained_weights))

net.snn.stdp_setup(None, None, positive_update=False, negative_update=False)

for syn in net.feature_synapses:
    syn.weight = ac_init.feature_weight

# live: ac_init.NEAR_BOX / FAR_BOX are pixel coords on the 1920x1080, 29.97 fps training footage, so the camera must match
frame_w, frame_h = 1920, 1080

if sys.platform == "darwin":
    cam_backend = cv2.CAP_AVFOUNDATION
elif os.name == "nt":
    cam_backend = cv2.CAP_DSHOW
else:
    cam_backend = cv2.CAP_V4L2


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cam_backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    if not cap.isOpened():
        raise RuntimeError(f"camera {camera_index} did not open.")
    return cap


def list_cameras(max_index=6):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cam_backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
            ret, frame = cap.read()
            size = f"{frame.shape[1]}x{frame.shape[0]}" if ret else "opened, no frame"
            print(f"camera {i}: {size} @ {cap.get(cv2.CAP_PROP_FPS):.2f} fps")
        cap.release()


def track(mask, box, min_area, max_area):
    x, y, w, h = box
    crop = mask[y:y+h, x:x+w]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(crop)
    if num_labels <= 1:
        return None, None
    best_i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    area = stats[best_i, cv2.CC_STAT_AREA]
    if not (min_area <= area <= max_area):
        return None, None
    top = stats[best_i, cv2.CC_STAT_TOP]
    height = stats[best_i, cv2.CC_STAT_HEIGHT]

    return top + height / 2, area

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y


def live_model(camera_index=video7, preview=False):
    cap = open_camera(video7)
    backSub = cv2.createBackgroundSubtractorMOG2()

    min_far_delta_away = 0.25
    max_far_delta_away = 1.5

    min_near_delta_away = -6.0
    max_near_delta_away = -1.0

    min_far_delta_torwards = -1.5
    max_far_delta_torwards = -0.25

    far_dy_scale = 4.0

    min_near_delta_torwards = 1.0
    max_near_delta_torwards = 6.9

    min_far_area = 10
    max_far_area = 100

    min_near_area = 100
    max_near_area = 10000

    far_diff_min = 4
    far_slope = 0.719

    near_diff_min = 24
    near_slope = 0.0913

    fps = cap.get(cv2.CAP_PROP_FPS)         # live: cameras often report 0 fps
    descent_window = int(45 * fps)

    frames_since_descent = None
    frames_since_vanish = None
    vanish_dead_zone = int(ac_init.vanish_dead_sec * fps)
    vanish_frames = int(len(net.vanished_layer) * fps)
    descent_streak = 0
    plane_gone_frames = int(ac_init.plane_gone_sec * fps)
    frames_confirmed_missing = 0
    recent_descent_frames = int(10 * fps)
    min_descent_frames = 5

    near_y_old = None
    far_y_old = None
    old_near_area = None
    old_far_area = None
    frame_idx = 0

    plane_seen = False
    plane_missing = False
    was_descending = False
    returned = False
    returned_frames = 0

    frames_plane_seen = 0
    min_frames_plane_seen = int(2 * fps)
    frames_plane_missing = 0
    min_frames_plane_missing = int(2 * fps)

    box = {b: {"seen": False, "was_seen": False, "missing": False, "n_seen": 0, "n_missing": 0, "n_confirmed": 0,
               "areas": []} for b in ("near", "far")}
    gone_layer = {"near": net.near_gone_layer, "far": net.far_gone_layer}
    returned_box = None

    decisions = {"LANDING": net.landing_decision[0].idx,
                 "TAKEOFF": net.takeoff_decision[0].idx,
                 "TOUCH-AND-GO": net.touch_go_decision[0].idx}
    counts = {name: 0 for name in decisions}

    fx, fy, fw, fh = ac_init.FAR_BOX
    nx, ny, nw, nh = ac_init.NEAR_BOX

    print(f"Live! Camera {camera_index} at {fps:.2f} fps.", flush=True)
    t_start = time.time()
    failed_reads = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1                            # live: webcams drop the first frames while warming up
            if failed_reads >= 30:                       # live: feed really dropped -> reconnect and keep running
                cap.release()
                time.sleep(1)
                try:
                    cap = open_camera(camera_index)
                except RuntimeError as e:
                    print(e, flush=True)
                failed_reads = 0
            else:
                time.sleep(0.03)
            continue
        failed_reads = 0
        if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
            frame = cv2.resize(frame, (frame_w, frame_h))  # live: put any camera size onto the 1080p grid the boxes use
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = backSub.apply(gray)
        near_mask = mask.copy()
        near_mask[fy:fy+fh, fx:fx+fw] = 0
        near_y, near_area = track(near_mask, ac_init.NEAR_BOX, min_near_area, max_near_area)
        far_y, far_area = track(mask, ac_init.FAR_BOX, min_far_area, max_far_area)

        is_blob = near_y is not None or far_y is not None

        for b, present, area in (("near", near_y is not None, near_area), ("far", far_y is not None, far_area)):
            st = box[b]
            if present:
                st["areas"] = (st["areas"] + [area])[-min_frames_plane_seen:]
                st["n_missing"] = 0
                st["n_seen"] += 1
                if st["n_seen"] > min_frames_plane_seen:
                    st["seen"], st["missing"] = True, False
            else:
                st["areas"] = []
                st["n_seen"] = 0
                st["n_missing"] += 1
                if st["n_missing"] > min_frames_plane_missing:
                    st["missing"], st["seen"] = True, False

        if is_blob:
            frames_plane_missing = 0
            frames_plane_seen += 1
            if frames_plane_seen > min_frames_plane_seen:
                plane_seen = True
                plane_missing = False
        else:
            frames_plane_seen = 0
            frames_plane_missing += 1
            if frames_plane_missing > min_frames_plane_missing:
                plane_missing = True
                plane_seen = False

        # live: spike time is relative to the SNN's "now", so every spike for this frame goes in at t=0
        if plane_seen:
            for n in net.blob_detected_layer:
                net.snn.add_spike(0, n, 13)

        descent_fired = False

        if plane_seen:
            if near_y is not None and near_y_old is not None:
                near_dy = get_slope(near_y_old, near_y)
                if min_near_delta_away < near_dy < max_near_delta_away:
                    descent_fired = True
                    for n in net.descent_near_layer:
                        net.snn.add_spike(0, n, abs(near_dy))

                elif min_near_delta_torwards < near_dy < max_near_delta_torwards:
                    for n in net.ascent_near_layer:
                        net.snn.add_spike(0, n, abs(near_dy))

                if near_area > min_near_area:
                    if old_near_area is None:
                        old_near_area = near_area
                    diff_near_area = near_area - old_near_area
                    old_near_area = near_area
                    if diff_near_area > near_diff_min:
                        value = 1 + (diff_near_area - near_diff_min) * near_slope
                        for n in net.grow_layer:
                            net.snn.add_spike(0, n, value, exist="add")
                    elif diff_near_area < -near_diff_min:
                        value = 1 + (-diff_near_area - near_diff_min) * near_slope
                        for n in net.shrink_layer:
                            net.snn.add_spike(0, n, value, exist="add")

            if far_y is not None and far_y_old is not None:
                far_dy = get_slope(far_y_old, far_y)
                if min_far_delta_away < far_dy < max_far_delta_away:
                    for n in net.ascent_far_layer:
                        net.snn.add_spike(0, n, abs(far_dy) * far_dy_scale)

                if min_far_delta_torwards < far_dy < max_far_delta_torwards:
                    for n in net.descent_far_layer:
                        net.snn.add_spike(0, n, abs(far_dy) * far_dy_scale)

                if far_area > 1:
                    if old_far_area is None:
                        old_far_area = far_area
                    diff_far_area = far_area - old_far_area
                    old_far_area = far_area
                    if diff_far_area > far_diff_min:
                        value = 1 + (diff_far_area - far_diff_min) * far_slope
                        for n in net.grow_layer:
                            net.snn.add_spike(0, n, value, exist='add')
                    elif diff_far_area < -far_diff_min:
                        value = 1 + (-diff_far_area - far_diff_min) * far_slope
                        for n in net.shrink_layer:
                            net.snn.add_spike(0, n, value, exist='add')

        if frames_since_descent is not None:
            frames_since_descent += 1
        if frames_since_vanish is not None:
            frames_since_vanish += 1

        if descent_fired:
            descent_streak += 1
        else:
            descent_streak = 0
        if descent_streak >= min_descent_frames:
            frames_since_descent = 0

        if box["near"]["seen"] and near_y is not None:
            if frames_since_descent is not None and frames_since_descent <= recent_descent_frames:
                was_descending = True
        elif box["near"]["missing"] and was_descending:
            frames_since_vanish = 0
            was_descending = False

        if frames_since_descent is not None and frames_since_descent < descent_window:
            for n in net.recent_descent_layer:
                net.snn.add_spike(0, n, 13)

        for b in ("near", "far"):
            st = box[b]
            if st["seen"] and not st["was_seen"]:
                plane_sized = b == "near" or np.median(st["areas"]) >= ac_init.far_return_min_area
                if frames_since_vanish is not None and vanish_dead_zone <= frames_since_vanish < vanish_frames and plane_sized:
                    returned = True
                    returned_box = b
                    returned_frames = 0
                    frames_since_vanish = None
            st["was_seen"] = st["seen"]
            st["n_confirmed"] = st["n_confirmed"] + 1 if st["missing"] else 0
            if st["n_confirmed"] == plane_gone_frames:
                for n in gone_layer[b]:
                    net.snn.add_spike(0, n, 13)
                if returned and returned_box == b:
                    returned = False
        frames_confirmed_missing = frames_confirmed_missing + 1 if plane_missing else 0
        if frames_confirmed_missing == plane_gone_frames:
            for n in net.plane_gone_layer:
                net.snn.add_spike(0, n, 13)
        if returned:
            returned_frames += 1
            if returned_frames > int(75 * fps):
                returned = False
        if returned:
            for n in net.returned_layer:
                net.snn.add_spike(0, n, 13)

        if frames_since_vanish is not None:
            sec = int(frames_since_vanish // fps)
            if sec < len(net.vanished_layer):
                net.snn.add_spike(0, net.vanished_layer[sec], 13)

        # live: advance the SNN exactly one frame, then read only that frame's spikes
        net.snn.simulate(time_steps=1)
        spikes = net.snn.ispikes[-1]
        for name, idx in decisions.items():
            if spikes[idx]:
                counts[name] += 1
                ops = counts["LANDING"] + counts["TAKEOFF"] + (2 * counts["TOUCH-AND-GO"])
                print(f"[{time.strftime('%H:%M:%S')} | t={frame_idx / fps:7.1f}s] {name:<12}  "
                      f"landings {counts['LANDING']}  takeoffs {counts['TAKEOFF']}  "
                      f"touch-and-goes {counts['TOUCH-AND-GO']}  operations {ops}", flush=True)
        # live: trim spike history so memory stays flat; shorten (not clear) keeps the last step so gate latches hold
        net.snn.shorten_spike_train()

        if frame_idx == int(10 * fps):                   # live: the SNN's timers count frames, so the camera must really run at fps
            real_fps = frame_idx / (time.time() - t_start)
            if abs(real_fps - fps) > 0.1 * fps:
                print(f"WARNING! camera is delivering {real_fps:.1f} fps, timers assume {fps:.2f}", flush=True)

        if preview:
            view = frame.copy()
            cv2.rectangle(view, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 3)
            cv2.rectangle(view, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
            cv2.putText(view, f"L: {counts['LANDING']}  T: {counts['TAKEOFF']}  TG: {counts['TOUCH-AND-GO']}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            try:
                cv2.imshow("ac_live", cv2.resize(view, (frame_w // 2, frame_h // 2)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print("preview failed; no windows")
                preview = False

        if near_y is None:
            old_near_area = None
        if far_y is None:
            old_far_area = None

        near_y_old = near_y
        far_y_old = far_y
        frame_idx += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    return counts

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        list_cameras()
        sys.exit()
    numbers = [a for a in args if a.isdigit()]
    camera_index = int(numbers[0]) if numbers else 0
    try:
        live_model(camera_index, preview="--no-preview" not in args)
    except KeyboardInterrupt:
        print("Video stopped.")
    except RuntimeError as e:
        print(e)

counts = live_model(camera_index=video7, preview=True)                  # In real testing, change to camera_index = 0; video8 for test

landings = counts["LANDING"]
takeoffs = counts["TAKEOFF"]
touch_gos = counts["TOUCH-AND-GO"]
raw_ops = landings + takeoffs + touch_gos
final_ops = landings + takeoffs + (2 * touch_gos)

print()
print("Video ended. Final counts:")
print("Landings:", landings)
print("Takeoffs:", takeoffs)
print("Touch & goes:", touch_gos)

print("Raw operations count:", raw_ops)
print("Final operations count:", final_ops)
