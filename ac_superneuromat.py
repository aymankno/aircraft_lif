### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

snn = SNN()

# 25 neurons each - eyes of SNN (input layers)
descent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

grow_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
shrink_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

descent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

grow_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
shrink_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

recent_descent_layer = [snn.create_neuron(threshold=0.5)]                           # one neuron because recent_descent is yes/no
vanished_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(20)]    # 20 neurons to make the layer into a timer, one neuron / sec

learning_inputs = descent_near_layer + ascent_near_layer +\
         descent_far_layer + ascent_far_layer + grow_near_layer + shrink_near_layer +\
         grow_far_layer + shrink_far_layer

memory_inputs = recent_descent_layer + vanished_layer

# 10 neurons each -  decision layers (output layers)
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

outputs = landing_layer + takeoff_layer + touch_go_layer

for pre in learning_inputs:                                       # Makes synapses to connect everything together
    for post in outputs:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

for pre in recent_descent_layer:
    for post in takeoff_layer:
        snn.create_synapse(pre, post, weight=-0.1, stdp_enabled=False)

for pre in vanished_layer:
    for post in touch_go_layer:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

NEAR_BOX = (468, 1, 967, 747)
FAR_BOX  = (974, 604, 265, 151)

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

### TRAINING PIPELINE BELOW
print()
print("TRAINING:")
video9_labels = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/video9_labels.csv"
pd.read_csv(video9_labels)


defined_outputs = {"takeoff": takeoff_layer,
                   "landing": landing_layer,
                   "touch/go": touch_go_layer}

defined_near_inputs = {"takeoff": ascent_near_layer,
                       "landing": descent_near_layer}

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses 
snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
cap = cv2.VideoCapture(video9)
backSub = cv2.createBackgroundSubtractorMOG2()

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
                snn.add_spike((f_idx + 1), n, 9)

min_far_delta = 0.25 # all footage was with traffic departing away from camera. Will add fix in future to account for operations torwards camera
min_near_delta = -6.0 # see changelog 10/9

max_far_delta = 1.5
max_near_delta = -1.0

max_near_area = 3000 # ! - may change; test max were very far from median & mean for both
max_far_area = 325 # !

min_far_area = 85
min_near_area = 850

far_diff_min = 4 # min change to count as a change & spike
far_diff_max = 20 # everything above max counts as same; fires whole layer
far_slope = 0.719

near_diff_min = 24
near_diff_max = 150
near_slope = 0.0913

fps = cap.get(cv2.CAP_PROP_FPS)
descent_window = int(18 * fps) # how long recent_descent until leaked to 0
min_descent_frames = 5         # descent frames in a row before counted (for noise)
vanished_unseen_min = 5        # frames it doesn't see aircraft before considered vanished
vanished_seen_min = 5          # frames it sees an aircraft before considering aircraft unvanished

frames_since_descent = None
frames_since_vanish = None
descent_streak = 0
frames_gone = 0
was_descending = False
frames_seen = 0

near_y_old = None
far_y_old = None
old_near_area = None
old_far_area = None
frame_idx = 0

tg_from = 9440
tg_to = 11388

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = backSub.apply(gray)
    near_mask = mask.copy()
    near_mask[604:755, 974:1239] = 0
    near_y, near_area = track(near_mask, NEAR_BOX, 1, 10000)
    far_y, far_area = track(mask, FAR_BOX, 1, 10000)

    descent_fired = False
    
    if near_y is not None and near_y_old is not None:
        near_dy = get_slope(near_y_old, near_y)
#        if near_dy > min_move:
#            print(near_dy)                                     # commented out until reversing is implemented (see changelog)
#            for n in ascent_near_layer:
#                snn.add_spike(frame_idx, n, abs(near_dy))
        if min_near_delta < near_dy < max_near_delta:
            descent_fired = True
            for n in descent_near_layer:
                snn.add_spike(frame_idx, n, abs(near_dy))

        if near_area > min_near_area:
            if old_near_area is None:
                old_near_area = near_area
            diff_near_area = near_area - old_near_area
            old_near_area = near_area
            if diff_near_area > near_diff_min:
                value = 1 + (diff_near_area - near_diff_min) * near_slope
                for n in grow_near_layer:
                    snn.add_spike(frame_idx, n, value)
            elif diff_near_area < -near_diff_min:
                value = 1 + (-diff_near_area - near_diff_min) * near_slope
                for n in shrink_near_layer:
                    snn.add_spike(frame_idx, n, value)



    if far_y is not None and far_y_old is not None:
        far_dy = get_slope(far_y_old, far_y)
        if min_far_delta < far_dy < max_far_delta:
            for n in ascent_far_layer:
                snn.add_spike(frame_idx, n, abs(far_dy))
#        elif far_dy < -min_move:
#            print(far_dy)
#            for n in descent_far_layer:                        # see changelog 10/9
#                snn.add_spike(frame_idx, n, abs(far_dy))

        if far_area > min_far_area:
            if old_far_area is None:
                old_far_area = far_area
            diff_far_area = far_area - old_far_area
            old_far_area = far_area
            if diff_far_area > far_diff_min:
                value = 1 + (diff_far_area - far_diff_min) * far_slope
                for n in grow_far_layer:
                    snn.add_spike(frame_idx, n, value)
            elif diff_far_area < -far_diff_min:
                value = 1 + (-diff_far_area - far_diff_min) * far_slope
                for n in shrink_far_layer:
                    snn.add_spike(frame_idx, n, value)

    if near_y is not None or far_y is not None:
        frames_seen += 1
    else:
        frames_seen = 0

    plane_seen = frames_seen >= vanished_seen_min
    plane_gone = frames_seen == 0

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

    if plane_seen:
        frames_gone = None
        was_descending = frames_since_descent is not None and frames_since_descent <= 3
    elif plane_gone and was_descending:
        frames_gone = 0 if frames_gone is None else frames_gone + 1
        if frames_gone >= vanished_unseen_min:
            frames_since_vanish = 0
            was_descending = False

    if frames_since_descent is not None and frames_since_descent < descent_window:
        for n in recent_descent_layer:
            snn.add_spike(frame_idx, n, 13)
    if frames_since_vanish is not None:
        sec = int(frames_since_vanish // fps)
        if sec < len(vanished_layer):
            snn.add_spike(frame_idx, vanished_layer[sec], 13) # spikes only the neuron for sec

    if near_y is None:
        old_near_area = None
    if far_y is None:
        old_far_area = None

    if tg_from <= frame_idx <= tg_to:
        print(frame_idx, near_y, far_y, descent_fired, descent_streak,
          frames_since_descent, was_descending, frames_seen, frames_gone)

    near_y_old = near_y
    far_y_old = far_y
    frame_idx += 1

print(frame_idx)

# Near
descent_near_ids = []
for n in descent_near_layer:                # SuperNeuroMAT looks at neurons as idx, so need ids for each
    descent_near_ids.append(n.idx)

ascent_near_ids = []
for neuron in ascent_near_layer:
    ascent_near_ids.append(neuron.idx)

grow_near_ids = []
for neuron in grow_near_layer:
    grow_near_ids.append(neuron.idx)

shrink_near_ids = []
for neuron in shrink_near_layer:
    shrink_near_ids.append(neuron.idx)

# Far
descent_far_ids = []
for neuron in descent_far_layer:
    descent_far_ids.append(neuron.idx)

ascent_far_ids = []
for neuron in ascent_far_layer:
    ascent_far_ids.append(neuron.idx)

grow_far_ids = []
for neuron in grow_far_layer:
    grow_far_ids.append(neuron.idx)

shrink_far_ids = []
for neuron in shrink_far_layer:
    shrink_far_ids.append(neuron.idx)

# Memory inputs:
vanish_ids = []
for neuron in vanished_layer:
    vanish_ids.append(neuron.idx)

# Outputs
landing_ids = []
for neuron in landing_layer:
    landing_ids.append(neuron.idx)

takeoff_ids = []
for neuron in takeoff_layer:
    takeoff_ids.append(neuron.idx)

touch_go_ids = []
for neuron in touch_go_layer:
    touch_go_ids.append(neuron.idx)

W = snn.weight_mat()

descent_landing_before = W[np.ix_(descent_near_ids[:11], landing_ids)].mean()
descent_takeoff_before = W[np.ix_(descent_near_ids[:11], takeoff_ids)].mean()

ascent_landing_before = W[np.ix_(ascent_far_ids[:2], landing_ids)].mean()
ascent_takeoff_before = W[np.ix_(ascent_far_ids[:2], takeoff_ids)].mean()

vanish_touch_go_before = W[np.ix_(vanish_ids, touch_go_ids)].mean()

steps = frame_idx
snn.simulate(time_steps=steps) # simulates full stdp training with teacher spikes
print("Vanish spikes:", snn.ispikes[:, vanish_ids].sum())
print("Takeoff teacher:", teacher_count["takeoff"], "actual:", snn.ispikes[:, takeoff_ids].sum())
print("Landing teachers", teacher_count["landing"], "actual:", snn.ispikes [:, landing_ids].sum())
print("Touch/go teacher:", teacher_count["touch/go"], "actual:", snn.ispikes[:, touch_go_ids].sum())

W = snn.weight_mat() # saves the NEW weights after training from training spikes

descent_landing_after = W[np.ix_(descent_near_ids[:11], landing_ids)].mean()
descent_takeoff_after = W[np.ix_(descent_near_ids[:11], takeoff_ids)].mean()

ascent_landing_after = W[np.ix_(ascent_far_ids[:2], landing_ids)].mean()
ascent_takeoff_after = W[np.ix_(ascent_far_ids[:2], takeoff_ids)].mean()

vanish_touch_go_after = W[np.ix_(vanish_ids, touch_go_ids)].mean()

print()
print("BEFORE:")
print("Descent-landing:", descent_landing_before)
print("Descent-takeoff:", descent_takeoff_before)
print("Ascent-landing:", ascent_landing_before)
print("Ascent-takeoff:", ascent_takeoff_before)
print("Vanish-touch/go:", vanish_touch_go_before)

print()


print("AFTER:")
print("Descent-landing:", descent_landing_after, "percent change:", ((descent_landing_after - descent_landing_before) / descent_landing_before) * 100)
print("Descent-takeoff:", descent_takeoff_after, "percent change:", ((descent_takeoff_after - descent_takeoff_before) / descent_takeoff_before) * 100)
print("Ascent-landing:", ascent_landing_after, "percent change:", ((ascent_landing_after - ascent_landing_before) / ascent_landing_before) * 100)
print("Ascent-takeoff:", ascent_takeoff_after, "percent change:", ((ascent_takeoff_after - ascent_takeoff_before) / ascent_takeoff_before) * 100)
print("Vanish-touch/go:", vanish_touch_go_after, "percent change:", ((vanish_touch_go_after - vanish_touch_go_before) / vanish_touch_go_before) * 100)
print()

### SYNTHETIC TESTING BELOW - NOT YET CHANGED TO VIDEO
print("TESTING:")
snn.stdp_setup(Apos, Aneg, positive_update=False, negative_update=False) # turn off stdp training; now in testing

test_start_takeoff = 18500 # new start/ends for takeoffs & landings
test_end_takeoff = 19062

test_start_landing = 20682
test_end_landing = 21176

test_log = [(test_start_takeoff, test_end_takeoff, "takeoff"),
            (test_start_landing, test_end_landing, "landing")]

for start, end, label in test_log:
    for frame in range(start, end):
        for n in defined_inputs[label]:
            snn.add_spike(frame-steps, n, train_values[label]) # train_values: lines 82/83

snn.simulate(time_steps=22000)

takeoff_takeoff_count = snn.ispikes[test_start_takeoff:test_end_takeoff, takeoff_ids].sum() # counts spikes from each group during test window
takeoff_landing_count = snn.ispikes[test_start_takeoff:test_end_takeoff, landing_ids].sum()
landing_landing_count = snn.ispikes[test_start_landing:test_end_landing, landing_ids].sum()
landing_takeoff_count = snn.ispikes[test_start_landing:test_end_landing, takeoff_ids].sum()

print("Takeoff-takeoff count:", takeoff_takeoff_count)
print("Takeoff-landing count:", takeoff_landing_count)
print("Landing-landing count:", landing_landing_count)
print("Landing-takeoff count:", landing_takeoff_count)
print()

### VIDEO TESTING AND TRAINING; SAME VIDEO
print("VIDEO TESTING BELOW:")
video2_online = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/videos/video2_online.mp4"
cap = cv2.VideoCapture(video2_online)

backSub = cv2.createBackgroundSubtractorMOG2()

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

frame_num = 0
y_old = None
video_start = snn.ispikes.shape[0]
delta_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = backSub.apply(gray)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        best_i = np.argmax(stats[1: , cv2.CC_STAT_AREA]) + 1
        if 1000 <= stats[best_i, cv2.CC_STAT_AREA] <= 25000:
            centroid = (stats[best_i, cv2.CC_STAT_TOP] + (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT])) / 2
            if y_old is None:
                y_old = centroid

            top_left = (stats[best_i, cv2.CC_STAT_LEFT], stats[best_i, cv2.CC_STAT_TOP])
            right = stats[best_i, cv2.CC_STAT_LEFT] + stats[best_i, cv2.CC_STAT_WIDTH]
            bottom_right = (right, stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT])
            delta_y = get_slope(y_old, centroid)
            y_old = centroid
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 20, cv2.LINE_8)

    snn.simulate(1)
    latest = snn.ispikes[-1]
    takeoff_spiked = latest[takeoff_ids].any()
    if takeoff_spiked:
        print("TAKEOFF")
    landing_spiked = latest[landing_ids].any()
    if landing_spiked:
        print("LANDING")
    cv2.imshow("SuperNeuroMAT SNN", frame)
    cv2.waitKey(33)
    frame_num += 1

 # snn.simulate(frame_num+50)

print()
print("Descent layer", snn.ispikes[video_start:(video_start+frame_num), descent_ids ].sum())
print("Ascent layer", snn.ispikes[video_start:(video_start+frame_num), ascent_ids ].sum())
print("Takeoff layer", snn.ispikes[video_start:(video_start+frame_num), takeoff_ids ].sum())
print("Landing layer", snn.ispikes[video_start:(video_start+frame_num), landing_ids ].sum())
print()
