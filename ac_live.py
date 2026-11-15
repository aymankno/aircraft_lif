import os
import sys
import time

import cv2
import numpy as np

import ac_init

here = os.path.dirname(os.path.abspath(__file__))
trained_weights = os.path.join(here, "trained_weights.npy")
video10 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/10muted.mp4"

min_far_area = 10
max_far_area = 100

min_near_area = 100
max_near_area = 10000

far_diff_min = 4
far_slope = 0.719

near_diff_min = 24
near_slope = 0.0913

frame_w, frame_h = 1920, 1080
FPS = ac_init.FPS
flush_sec = 45

if sys.platform == "darwin":
    cam_backend = cv2.CAP_AVFOUNDATION
elif os.name == "nt":
    cam_backend = cv2.CAP_DSHOW
else:
    cam_backend = cv2.CAP_V4L2

def get_mins(steps):
    secs = steps / FPS
    if secs >= 60:
        mins = int(secs / 60)
        secs = int(secs % 60)
        timed = f"{mins}:{secs}"
    else:
        timed = f"{secs}s"
    return timed

def frames(sec):
    return int(sec * FPS)


def load_network(weights_path=trained_weights):
    net = ac_init.build_network()
    W = np.load(weights_path)
    n = net.snn.num_neurons
    if W.shape != (n, n):
        raise RuntimeError(f"{os.path.basename(weights_path)} is {W.shape[0]}x{W.shape[1]} but ac_init builds {n} neurons; "
                           "ac_init changed since training, re-run ac_train.")
    net.snn.set_weights_from_mat(W)
    net.snn.stdp_setup(None, None, positive_update=False, negative_update=False)
    for syn in net.feature_synapses:
        syn.weight = ac_init.feature_weight
    return net


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cam_backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise RuntimeError(f"camera {camera_index} did not open.")
    return cap


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"video {path} did not open.")
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


class FrameEncoder:
    def __init__(self, net):
        self.net = net
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.forget_motion()
        net.snn.add_spike(0, net.plane_absent, 1.0)

    def forget_motion(self):
        self.near_y_old = None
        self.far_y_old = None
        self.old_near_area = None
        self.old_far_area = None

    def encode(self, frame):
        net, snn = self.net, self.net.snn
        fx, fy, fw, fh = ac_init.FAR_BOX

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.backSub.apply(gray)
        near_mask = mask.copy()
        near_mask[fy:fy+fh, fx:fx+fw] = 0
        near_y, near_area = track(near_mask, ac_init.NEAR_BOX, min_near_area, max_near_area)
        far_y, far_area = track(mask, ac_init.FAR_BOX, min_far_area, max_far_area)

        is_blob = float(near_y is not None or far_y is not None)
        snn.add_spike(0, net.plane_present_count, is_blob)
        snn.add_spike(0, net.plane_missing_count, 1.0 - is_blob)
        for present_count, missing_count, y in ((net.near_present_count, net.near_missing_count, near_y),
                                                (net.far_present_count, net.far_missing_count, far_y)):
            present = float(y is not None)
            snn.add_spike(0, present_count, present)
            snn.add_spike(0, missing_count, 1.0 - present)
        if far_y is not None:
            snn.add_spike(0, net.far_big, far_area)

        for y, y_old, scale, descent, ascent in ((near_y, self.near_y_old, 1.0, "descent_near", "ascent_near"),
                                                 (far_y, self.far_y_old, ac_init.far_dy_scale, "descent_far", "ascent_far")):
            if y is not None and y_old is not None:
                value = (y - y_old) * scale
                if descent == "descent_near":
                    for n in net.down_steps:
                        snn.add_spike(0, n, value)
                    for n in net.up_steps:
                        snn.add_spike(0, n, -value)
                snn.add_spike(0, net.veto[descent], value)
                snn.add_spike(0, net.veto[ascent], -value)
                for n in net.thermometers[descent]:
                    snn.add_spike(1, n, value)
                for n in net.thermometers[ascent]:
                    snn.add_spike(1, n, -value)

        # grow / shrink (left as python for now)
        if near_y is not None and self.near_y_old is not None:
            if self.old_near_area is None:
                self.old_near_area = near_area
            diff_near_area = near_area - self.old_near_area
            self.old_near_area = near_area
            if diff_near_area > near_diff_min:
                value = 1 + (diff_near_area - near_diff_min) * near_slope
                for n in net.grow_layer:
                    snn.add_spike(0, n, value, exist="add")
            elif diff_near_area < -near_diff_min:
                value = 1 + (-diff_near_area - near_diff_min) * near_slope
                for n in net.shrink_layer:
                    snn.add_spike(0, n, value, exist="add")

        if far_y is not None and self.far_y_old is not None:
            if self.old_far_area is None:
                self.old_far_area = far_area
            diff_far_area = far_area - self.old_far_area
            self.old_far_area = far_area
            if diff_far_area > far_diff_min:
                value = 1 + (diff_far_area - far_diff_min) * far_slope
                for n in net.grow_layer:
                    snn.add_spike(0, n, value, exist="add")
            elif diff_far_area < -far_diff_min:
                value = 1 + (-diff_far_area - far_diff_min) * far_slope
                for n in net.shrink_layer:
                    snn.add_spike(0, n, value, exist="add")

        if near_y is None:
            self.old_near_area = None
        if far_y is None:
            self.old_far_area = None

        self.near_y_old = near_y
        self.far_y_old = far_y
        return near_y, far_y


def live_model(source=0, preview=False, flush=True, net=None):
    if net is None:
        net = load_network()
    snn = net.snn

    is_file = isinstance(source, str)
    cap = open_video(source) if is_file else open_camera(source)
    encoder = FrameEncoder(net)

    decisions = {"LANDING": net.landing_decision[0].idx,
                 "TAKEOFF": net.takeoff_decision[0].idx,
                 "TOUCH-AND-GO": net.touch_go_decision[0].idx}
    counts = {name: 0 for name in decisions}
    step = 0

    def read_decisions(spikes, at_step):
        for name, idx in decisions.items():
            if spikes[idx]:
                counts[name] += 1
                ops = counts["LANDING"] + counts["TAKEOFF"] + (2 * counts["TOUCH-AND-GO"])
                print(f"{time.strftime('%H:%M:%S')} (t={get_mins(at_step)}): {name:<12}  "
                      f"landings {counts['LANDING']}  takeoffs {counts['TAKEOFF']}  "
                      f"touch-and-goes {counts['TOUCH-AND-GO']}  operations {ops}", flush=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if is_file:
        print(f"Replaying {os.path.basename(source)} ({src_fps:.2f} fps) through the live loop.", flush=True)
        if src_fps and abs(src_fps - FPS) > 0.1 * FPS:
            print(f"WARNING! file is {src_fps:.2f} fps, timers assuming {FPS:.2f}", flush=True)
    else:
        print(f"Live! Camera {source} reports {src_fps:.2f} fps; SNN steps at {FPS:.2f}. 'q' or Ctrl-C to stop.", flush=True)

    fx, fy, fw, fh = ac_init.FAR_BOX
    nx, ny, nw, nh = ac_init.NEAR_BOX

    period = 1.0 / FPS
    cam_interval = None
    n_intervals = 0
    last_read = None
    keep_acc = 0.0
    rate_window = frames(30)
    window_start, window_step = time.perf_counter(), 0
    failed_reads = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    break
                failed_reads += 1
                if failed_reads >= 30:
                    cap.release()
                    time.sleep(1)
                    try:
                        cap = open_camera(source)
                    except RuntimeError as e:
                        print(e, flush=True)
                    encoder.forget_motion()
                    last_read = None
                    failed_reads = 0
                else:
                    time.sleep(0.03)
                continue
            failed_reads = 0

            if not is_file:
                now = time.perf_counter()
                if last_read is not None and now - last_read < 0.1:
                    n_intervals += 1
                    alpha = max(0.05, 1.0 / n_intervals)
                    dt = now - last_read
                    cam_interval = dt if cam_interval is None else (1 - alpha) * cam_interval + alpha * dt
                last_read = now
                if cam_interval is not None and cam_interval < 0.85 * period:
                    keep_acc += cam_interval / period
                    if keep_acc < 1.0:
                        continue
                    keep_acc -= 1.0

            if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                frame = cv2.resize(frame, (frame_w, frame_h))

            near_y, far_y = encoder.encode(frame)

            snn.simulate(time_steps=1)
            read_decisions(snn.spike_train[-1], step)
            snn.shorten_spike_train()
            step += 1

            if not is_file and step - window_step >= rate_window:
                real_fps = (step - window_step) / (time.perf_counter() - window_start)
                if abs(real_fps - FPS) > 0.1 * FPS:
                    print(f"WARNING! only {real_fps:.1f} frames/s reaching the SNN, timers assume {FPS:.2f}", flush=True)
                window_start, window_step = time.perf_counter(), step

            if preview:
                view = frame.copy()
                cv2.rectangle(view, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 3)
                cv2.rectangle(view, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
                if near_y is not None:
                    cv2.line(view, (nx, int(ny + near_y)), (nx + nw, int(ny + near_y)), (0, 255, 0), 1)
                if far_y is not None:
                    cv2.line(view, (fx, int(fy + far_y)), (fx + fw, int(fy + far_y)), (0, 0, 255), 1)
                cv2.putText(view, f"L: {counts['LANDING']}  T: {counts['TAKEOFF']}  TG: {counts['TOUCH-AND-GO']}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                try:
                    cv2.imshow("ac_live", cv2.resize(view, (frame_w // 2, frame_h // 2)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("preview failed; no windows", flush=True)
                    preview = False
    except KeyboardInterrupt:
        print("Video stopped.", flush=True)
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    if flush:
        print(f"Flushing {flush_sec} s of SNN time . . .", flush=True)
        start = len(snn.spike_train)
        snn.simulate(time_steps=frames(flush_sec))
        for k, spikes in enumerate(snn.spike_train[start:]):
            read_decisions(spikes, step + k)

    return counts


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        list_cameras()
        sys.exit()

    paths = [a for a in args if not a.startswith("--") and not a.isdigit()]
    numbers = [a for a in args if a.isdigit()]
    source = paths[0] if paths else (int(numbers[0]) if numbers else 0)

    try:
        counts = live_model(video10, preview="--no-preview" not in args, flush="--no-flush" not in args)             # CHOOSE VIDEO / WEBCAM HERE
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    landings = counts["LANDING"]
    takeoffs = counts["TAKEOFF"]
    touch_gos = counts["TOUCH-AND-GO"]

    print()
    print("Final counts:")
    print("Landings:", landings)
    print("Takeoffs:", takeoffs)
    print("Touch & goes:", touch_gos)
    print("Raw operations count:", landings + takeoffs + touch_gos)
    print("Final operations count:", landings + takeoffs + (2 * touch_gos))   # In official reporting, touch & goes counts as 2 movements
