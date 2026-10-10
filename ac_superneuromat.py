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
size_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

descent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
size_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

inputs = descent_near_layer + ascent_near_layer + descent_far_layer + ascent_far_layer + size_near_layer + size_far_layer

# 10 neurons each -  decision layers (output layers)
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

outputs = landing_layer + takeoff_layer + touch_go_layer

for pre in inputs:                                       # Makes 4500 synapses to connect everything together
    for post in outputs:
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

def size_changing(centroid_y, centroid_x):
    pass # will write

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
cap = cv2.VideoCapture(video9)
backSub = cv2.createBackgroundSubtractorMOG2()

f_idx = 0
with open(video9_labels) as f:
    next(f)
    for row in f:                                    # Teacher Spike:
        row = row.strip()
        if not row:
            continue
        start, end, area, movement = row.split(',')
        start, end = int(start), int(end)
        for f_idx in range(start, end, 10):
            for n in defined_outputs[movement]:
                snn.add_spike((f_idx + 1), n, 10)

near_dys = []
far_dys = []
min_move = 0.1

max_near_area = 3000 # ! - may change; test max were very far from median & mean for both
max_far_area = 325 # !

min_far_area = 85
min_near_area = 850

near_y_old = None
far_y_old = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = backSub.apply(gray)
    near_mask = mask.copy()
    near_mask[604:755, 974:1239] = 0
    near_y, near_area = track(near_mask, NEAR_BOX, min_near_area, max_near_area)
    far_y, far_area = track(mask, FAR_BOX, min_far_area, max_far_area)
    
    if near_y is not None and near_y_old is not None:
        near_dy = get_slope(near_y_old, near_y)
        if near_dy > min_move:
            near_dys.append(near_dy)
            print(near_dy)
            for n in ascent_near_layer:
                snn.add_spike(frame_idx, n, abs(near_dy))
        elif near_dy < -min_move:
            near_dys.append(near_dy)
            print(near_dy)
            for n in descent_near_layer:
                snn.add_spike(frame_idx, n, abs(near_dy))
        if near_area > min_near_area:
            for n in size_near_layer:
                snn.add_spike(frame_idx, n, near_area)


    if far_y is not None and far_y_old is not None:
        far_dy = get_slope(far_y_old, far_y)
        if far_dy > min_move:
            far_dys.append(far_dy)
            print(far_dy)
            for n in ascent_far_layer:
                snn.add_spike(frame_idx, n, abs(far_dy))
        elif far_dy < -min_move:
            far_dys.append(far_dy)
            print(far_dy)
            for n in descent_far_layer:
                snn.add_spike(frame_idx, n, abs(far_dy))
        if far_area > min_far_area:
            for n in size_far_layer:
                snn.add_spike(frame_idx, n, far_area)

    near_y_old = near_y
    far_y_old = far_y
    frame_idx += 1

far_dys = np.asarray(far_dys)
near_dys = np.asarray(near_dys)

pos_far = far_dys[far_dys > 0.1]
neg_far = far_dys[far_dys < -0.1]

pos_near = near_dys[near_dys > 0.1]
neg_near = near_dys[near_dys < -0.1]


print()
print(f"FAR deltas: min: {np.min(far_dys)}, max: {np.max(far_dys)}, median (pos): {np.median(pos_far)}, mean (pos): {np.mean(pos_far)}, median (neg): {np.median(neg_far)}, mean (neg): {np.mean(neg_far)}")
print(f"NEAR deltas: min: {np.min(near_dys)}, max: {np.max(near_dys)}, median (pos): {np.median(pos_near)}, mean (pos): {np.mean(pos_near)}, median (neg): {np.median(neg_near)}, mean (neg): {np.mean(neg_near)}")
print()

descent_near_ids = []
for n in descent_near_layer:                # SuperNeuroMAT looks at neurons as idx, so need ids for each
    descent_near_ids.append(n.idx)

ascent_near_ids = []
for neuron in ascent_near_layer:
    ascent_near_ids.append(neuron.idx)

descent_far_ids = []
for neuron in descent_far_layer:                # SuperNeuroMAT looks at neurons as idx, so need ids for each
    descent_far_ids.append(neuron.idx)

ascent_far_ids = []
for neuron in ascent_far_layer:
    ascent_far_ids.append(neuron.idx)


landing_ids = []
for neuron in landing_layer:
    landing_ids.append(neuron.idx)

takeoff_ids = []
for neuron in takeoff_layer:
    takeoff_ids.append(neuron.idx)

# BELOW NOT YET TESTED / CHANGED

descent_landing_before = W[np.ix_(descent_ids[:8], landing_ids)].mean()
descent_takeoff_before = W[np.ix_(descent_ids[:2], takeoff_ids)].mean()

ascent_landing_before = W[np.ix_(ascent_ids[:8], landing_ids)].mean()
ascent_takeoff_before = W[np.ix_(ascent_ids[:2], takeoff_ids)].mean()

steps = 15520
snn.simulate(time_steps=steps) # simulates full stdp training with teacher spikes

W = snn.weight_mat() # saves the NEW weights after training from training spikes

descent_landing_after = W[np.ix_(descent_ids[:8], landing_ids)].mean()
descent_takeoff_after = W[np.ix_(descent_ids[:2], takeoff_ids)].mean()

ascent_landing_after = W[np.ix_(ascent_ids[:8], landing_ids)].mean()
ascent_takeoff_after = W[np.ix_(ascent_ids[:2], takeoff_ids)].mean()

print()
print("BEFORE:")
print("Descent-landing:", descent_landing_before)
print("Descent-takeoff:", descent_takeoff_before)
print("Ascent-landing:", ascent_landing_before)
print("Ascent-takeoff:", ascent_takeoff_before)

print()


print("AFTER:")
print("Descent-landing:", descent_landing_after, "DIFFERENCE", descent_landing_after - descent_landing_before)
print("Descent-takeoff:", descent_takeoff_after, "DIFFERENCE", descent_takeoff_after - descent_takeoff_before)
print("Ascent-landing:", ascent_landing_after, "DIFFERENCE", ascent_landing_after - ascent_landing_before)
print("Ascent-takeoff:", ascent_takeoff_after, "DIFFERENCE", ascent_takeoff_after - ascent_takeoff_before)
print()

### SYNTHETIC TESTING BELOW
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
