import cv2
import numpy as np
import ac_init

video1 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/1muted.mp4'
video2 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/2muted.mp4'
video3 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/3muted.mp4'
video4 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/4muted.mp4'
video5 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/5muted.mp4'
video6 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/6muted.mp4'
video7 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/7muted.mp4'
video8 = '/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/8muted.mp4'
video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
video10 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/10muted.mp4"

video9_labels = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/video9_labels.csv"
video10_labels = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/video10_labels.csv"

net = ac_init.build_network()

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

defined_outputs = {"takeoff": net.takeoff_layer,
                   "landing": net.landing_layer,
                   "touch/go": net.touch_go_layer}

defined_directions = {"away": net.traffic_away_layer, 
                      "torwards": net.traffic_torwards_layer}

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses; not used
net.snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

teacher_count = {"takeoff": 0, "landing": 0, "touch/go": 0}
with open(video9_labels) as f:
    next(f)
    for row in f:                                    # Teacher Spike:
        row = row.strip()
        if not row:
            continue
        start, end, area, movement, direction = [x.strip() for x in row.split(',')]
        start, end = int(start), int(end)
        for f_idx in range(start, end, 11):
            teacher_count[movement] += 1
            for n in defined_outputs[movement]:
                net.snn.add_spike((f_idx + 1), n, 9)
            for n in defined_directions[direction]:
                net.snn.add_spike((f_idx + 1), n, 9)

def run_forward(video):
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        yield frame

def run_model(video):
    source = run_forward(video)
    backSub = cv2.createBackgroundSubtractorMOG2()

    min_far_area = 10
    max_far_area = 100

    min_near_area = 100
    max_near_area = 10000

    far_diff_min = 4 # min change to count as a change & spike
    far_slope = 0.719

    near_diff_min = 24
    near_slope = 0.0913

    near_y_old = None
    far_y_old = None
    old_near_area = None
    old_far_area = None
    frame_idx = 0

    net.snn.add_spike(0, net.plane_absent, 1.0)                                    # plane_absent starts on, so plane_seen starts off

    for frame in source:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = backSub.apply(gray)
        near_mask = mask.copy()
        near_mask[604:755, 974:1239] = 0
        near_y, near_area = track(near_mask, ac_init.NEAR_BOX, min_near_area, max_near_area)
        far_y, far_area = track(mask, ac_init.FAR_BOX, min_far_area, max_far_area)

        # presence: raw "blob this frame" facts in, the presence() circuits do the debouncing
        is_blob = float(near_y is not None or far_y is not None)
        net.snn.add_spike(frame_idx, net.plane_present_count, is_blob)
        net.snn.add_spike(frame_idx, net.plane_missing_count, 1.0 - is_blob)
        for present_count, missing_count, y in ((net.near_present_count, net.near_missing_count, near_y),
                                                (net.far_present_count, net.far_missing_count, far_y)):
            present = float(y is not None)
            net.snn.add_spike(frame_idx, present_count, present)
            net.snn.add_spike(frame_idx, missing_count, 1.0 - present)
        if far_y is not None:
            net.snn.add_spike(frame_idx, net.far_big, far_area)

        for y, y_old, scale, descent, ascent in ((near_y, near_y_old, 1.0, "descent_near", "ascent_near"),
                                                 (far_y, far_y_old, ac_init.far_dy_scale, "descent_far", "ascent_far")):
            if y is not None and y_old is not None:
                value = (y - y_old) * scale                                 # = -dy * scale: positive when moving down the screen
                if descent == "descent_near":
                    for n in net.down_steps:
                        net.snn.add_spike(frame_idx, n, value)
                    for n in net.up_steps:
                        net.snn.add_spike(frame_idx, n, -value)
                net.snn.add_spike(frame_idx, net.veto[descent], value)
                net.snn.add_spike(frame_idx, net.veto[ascent], -value)
                for n in net.thermometers[descent]:
                    net.snn.add_spike(frame_idx + 1, n, value)                  # +1 frame so it lands with its veto
                for n in net.thermometers[ascent]:
                    net.snn.add_spike(frame_idx + 1, n, -value)

        # grow / shrink (left as python for now)
        if near_y is not None and near_y_old is not None:
            if old_near_area is None:
                old_near_area = near_area
            diff_near_area = near_area - old_near_area
            old_near_area = near_area
            if diff_near_area > near_diff_min:
                value = 1 + (diff_near_area - near_diff_min) * near_slope
                for n in net.grow_layer:
                    net.snn.add_spike(frame_idx, n, value, exist="add") # set all to exist = add for further "certainty", expierimental
            elif diff_near_area < -near_diff_min:
                value = 1 + (-diff_near_area - near_diff_min) * near_slope
                for n in net.shrink_layer:
                    net.snn.add_spike(frame_idx, n, value, exist="add")

        if far_y is not None and far_y_old is not None:
            if old_far_area is None:
                old_far_area = far_area
            diff_far_area = far_area - old_far_area
            old_far_area = far_area
            if diff_far_area > far_diff_min:
                value = 1 + (diff_far_area - far_diff_min) * far_slope
                for n in net.grow_layer:
                    net.snn.add_spike(frame_idx, n, value, exist='add')
            elif diff_far_area < -far_diff_min:
                value = 1 + (-diff_far_area - far_diff_min) * far_slope
                for n in net.shrink_layer:
                    net.snn.add_spike(frame_idx, n, value, exist='add')

        if near_y is None:
            old_near_area = None
        if far_y is None:
            old_far_area = None

        near_y_old = near_y
        far_y_old = far_y
        frame_idx += 1

    return frame_idx
    
frame_idx = run_model(video9)
net.snn.simulate(time_steps=frame_idx) # simulates full stdp training with teacher spike
np.save("trained_weights.npy", net.snn.weight_mat())
print("Training completed, weights saved.")