### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

snn = SNN()

# 25 neurons each - eyes of SNN (input layers)
descent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
growing_layer = [snn.create_neuron(threshold = ((i+1) * 0.5)) for i in range(25)]
shrinking_layer = [snn.create_neuron(threshold = ((i+1) * 0.5)) for i in range(25)]

inputs = descent_layer + ascent_layer

# 10 neurons each -  decision layers (output layers)
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

outputs = landing_layer + takeoff_layer + touch_go_layer

for pre in inputs:                                       # Makes 1500 synapses to connect everything together
    for post in outputs:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

### TRAINING PIPELINE BELOW
print()
print("TRAINING:")
video9_labels = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/video9_labels.csv"
pd.read_csv(video9_labels)


defined_outputs = {"takeoff": takeoff_layer,
                   "landing": landing_layer,
                   "touch/go": touch_go_layer}

for start, end, area, movement in video9_labels:                      # Teacher Spike:
    for frame in range(start, end, 10):             # Points to an event, pushes it to desired layer.
        for n in defined_outputs[movement]:
            snn.add_spike((frame + 1), n, 10)

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses
snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

defined_inputs = {"takeoff": ascent_layer,
                  "landing": descent_layer}

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

def size_changing(centroid_y, centroid_x):
    pass # will write

y_old = None

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"
cap = cv2.read(video9)
backSub = cv2.createBackgroundSubtractorMOG2()

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

    for start, end, near, movement in video9_labels:                       # Input Spike:
        for frame in range(start, end):                # Fake camera for when testing inputs with dummy values.
            for n in defined_inputs[near]:
                snn.add_spike(frame, n, abs(delta_y))

descent_ids = []
for neuron in descent_layer:                # SuperNeuroMAT looks at neurons as idx, so need ids for each
    descent_ids.append(neuron.idx)

ascent_ids = []
for neuron in ascent_layer:
    ascent_ids.append(neuron.idx)

landing_ids = []
for neuron in landing_layer:
    landing_ids.append(neuron.idx)

takeoff_ids = []
for neuron in takeoff_layer:
    takeoff_ids.append(neuron.idx)

W = snn.weight_mat() # before; what started as

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
descent_deltas = []
ascent_deltas = []

descent_parameters = within_parameters(-25, -0.1)
ascent_parameters = within_parameters(0.1, 15)


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
