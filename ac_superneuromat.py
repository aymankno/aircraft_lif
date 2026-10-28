### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np
import csv

rng = np.random.default_rng(0)

snn = SNN()

# 25 neurons each - eyes of SNN (input layers)
descent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

descent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

grow_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
shrink_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

recent_descent_layer = [snn.create_neuron(threshold=0.5)]                           # one neuron because recent_descent is yes/no
vanished_layer = [snn.create_neuron(threshold=(0.5)) for i in range(45)]    # 45 neurons to make the layer into a timer, one neuron / sec

blob_detected_layer = [snn.create_neuron(threshold=0.5)]

learning_inputs = descent_near_layer + ascent_near_layer +\
         descent_far_layer + ascent_far_layer

sizing_inputs = grow_layer + shrink_layer

memory_inputs = recent_descent_layer + vanished_layer + blob_detected_layer

# 10 neurons each - output layers
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

traffic_away_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
traffic_torwards_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

aircraft_detected_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

event_outputs = landing_layer + takeoff_layer + touch_go_layer # + aircraft_detected_layer

sizing_outputs = traffic_away_layer + traffic_torwards_layer

# Decision layers
landing_decision = [snn.create_neuron(threshold=1, leak=0.1, reset_state=0)]       # threshold will change based on tests for optimal cthresh
takeoff_decision = [snn.create_neuron(threshold=1, leak=0.1, reset_state=0)]
touch_go_decision = [snn.create_neuron(threshold=1, leak=0.1, reset_state=0)]

traffic_away_decision = [snn.create_neuron(threshold=1)]
traffic_torwards_decision = [snn.create_neuron(threshold=1)]

aircraft_detected_decision = [snn.create_neuron(threshold=1)]

decision_neurons = landing_decision + takeoff_decision + touch_go_decision + traffic_away_decision + traffic_torwards_decision

output_decisions = [(landing_layer, landing_decision),
                    (takeoff_layer, takeoff_decision),
                    (touch_go_layer, touch_go_decision),
                    (traffic_away_layer, traffic_away_decision),
                    (traffic_torwards_layer, traffic_torwards_decision),
                    (aircraft_detected_layer, aircraft_detected_decision)]

for layer, decision in output_decisions:
    for pre in layer:
        for post in decision:
            snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

for pre in learning_inputs:
    for post in event_outputs:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

for pre in sizing_inputs:
    for post in sizing_outputs:
        snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

for pre in recent_descent_layer:
    for post in takeoff_layer:
        snn.create_synapse(pre, post, weight=-0.1, stdp_enabled=False)    # must not exceed output leak (0.1), or state sinks without bound

for pre in vanished_layer:
    for post in landing_layer:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

for pre in blob_detected_layer:
    for post in aircraft_detected_layer:
        snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

# Feature neurons: fixed logic (no STDP, leak=inf so each is a same-frame detector)
# takeoff  = ascent AND NOT recent_descent
# touch/go = ascent AND recent_descent
takeoff_feat = snn.create_neuron(threshold=0.5)
touch_go_feat = snn.create_neuron(threshold=1.0)

ascent_any = [ascent_near_layer[0], ascent_far_layer[0]]     # any ascent this frame
for pre in ascent_any:
    snn.create_synapse(pre, takeoff_feat, weight=0.6, stdp_enabled=False)    # 1 ascent alone fires takeoff_feat
    snn.create_synapse(pre, touch_go_feat, weight=0.4, stdp_enabled=False)   # 2 ascents = 0.8, not enough alone

for pre in recent_descent_layer:
    snn.create_synapse(pre, takeoff_feat, weight=-2.0, stdp_enabled=False)
    snn.create_synapse(pre, touch_go_feat, weight=0.7, stdp_enabled=False) 

for feat, layer in [(takeoff_feat, takeoff_layer), (touch_go_feat, touch_go_layer)]:
    for post in layer:
        snn.create_synapse(feat, post, weight=0.6, stdp_enabled=True)

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

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

### TRAINING BELOW
print()
print("TRAINING:")
video9_labels = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/video9_labels.csv"


defined_outputs = {"takeoff": takeoff_layer,
                   "landing": landing_layer,
                   "touch/go": touch_go_layer}

defined_directions = {"away": traffic_away_layer, 
                      "torwards": traffic_torwards_layer}

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses; not used
snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

video9 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/9muted.mp4"

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
            for n in defined_directions[direction]:
                snn.add_spike((f_idx + 1), n, 9)

def run_model(video):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    backSub = cv2.createBackgroundSubtractorMOG2()

    min_far_delta_away = 0.25        # away
    max_far_delta_away = 1.5         # away
    
    min_near_delta_away = -6.0       # away
    max_near_delta_away = -1.0       # away

    min_far_delta_torwards = -1.5
    max_far_delta_torwards = -0.25

    far_dy_scale = 4.0               # far |dy| (0.25-1.5, moves in 0.5 px steps) -> near's 1-6 range, so far motion fires the thermometer layers

    min_near_delta_torwards = 1.0
    max_near_delta_torwards = 6.9

    min_far_area = 1
    max_far_area = 100

    min_near_area = 100
    max_near_area = 10000

    far_diff_min = 4 # min change to count as a change & spike
    far_slope = 0.719

    near_diff_min = 24
    near_slope = 0.0913

    fps = cap.get(cv2.CAP_PROP_FPS)
    descent_window = int(45 * fps) # how long recent_descent until leaked to 0
    video_start = 0

    frames_since_descent = None
    frames_since_vanish = None
    descent_streak = 0
    was_descending = False
    recent_descent_frames = int(10 * fps)
    min_descent_frames = 5

    near_y_old = None
    far_y_old = None
    old_near_area = None
    old_far_area = None
    frame_idx = 0

    plane_seen = False
    frames_plane_seen = 0
    min_frames_plane_seen = int(2 * fps)
    frames_plane_missing = 0
    min_frames_plane_missing = int(2 * fps)
    plane_missing = False
    frame_first_seen = None
    frame_disappeared = None

    with open(f'video9_aircraft_frames.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end"])

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

            is_blob = near_y is not None or far_y is not None

            if is_blob:
                frames_plane_missing = 0
                frame_disappeared = None
                frames_plane_seen += 1
                if frames_plane_seen == min_frames_plane_seen and frame_first_seen is None:
                    frame_first_seen = frame_idx
                elif frames_plane_seen > min_frames_plane_seen:
                    plane_seen = True
                    plane_missing = False
            else:
                frames_plane_seen = 0
                if frames_plane_missing == 0 and frame_disappeared is None:
                    frame_disappeared = frame_idx
                frames_plane_missing += 1
                if frames_plane_missing == min_frames_plane_missing and frame_first_seen is not None:
                    writer.writerow([frame_first_seen, frame_disappeared])
                    frame_first_seen = None
                    frame_disappeared = None

                elif frames_plane_missing > min_frames_plane_missing:
                    plane_missing = True
                    plane_seen = False

            if plane_seen:
                for n in blob_detected_layer:
                    snn.add_spike(frame_idx, n, 13)

            descent_fired = False

            if plane_seen:
                if near_y is not None and near_y_old is not None:
                    near_dy = get_slope(near_y_old, near_y)
                    if min_near_delta_away < near_dy < max_near_delta_away:
                        descent_fired = True
                        for n in descent_near_layer:                                        # away
                            snn.add_spike(frame_idx, n, abs(near_dy))

                    elif  min_near_delta_torwards < near_dy < max_near_delta_torwards:
                        for n in ascent_near_layer:                                         # torwards
                            snn.add_spike(frame_idx, n, abs(near_dy))

                    if near_area > min_near_area:
                        if old_near_area is None:
                            old_near_area = near_area
                        diff_near_area = near_area - old_near_area
                        old_near_area = near_area
                        if diff_near_area > near_diff_min:
                            value = 1 + (diff_near_area - near_diff_min) * near_slope
                            for n in grow_layer:
                                snn.add_spike(frame_idx, n, value, exist="add") # set all to exist = add for further "certainty", expierimental
                        elif diff_near_area < -near_diff_min:
                            value = 1 + (-diff_near_area - near_diff_min) * near_slope
                            for n in shrink_layer:
                                snn.add_spike(frame_idx, n, value, exist="add")



                if far_y is not None and far_y_old is not None:
                    far_dy = get_slope(far_y_old, far_y)
                    if min_far_delta_away < far_dy < max_far_delta_away:
                        for n in ascent_far_layer:                              # away
                            snn.add_spike(frame_idx, n, abs(far_dy) * far_dy_scale)

                    if min_far_delta_torwards < far_dy < max_far_delta_torwards:
                        descent_fired = True
                        for n in descent_far_layer:                             # torwards
                            snn.add_spike(frame_idx, n, abs(far_dy) * far_dy_scale)

                    if far_area > 1:
                        if old_far_area is None:
                            old_far_area = far_area
                        diff_far_area = far_area - old_far_area
                        old_far_area = far_area
                        if diff_far_area > far_diff_min:
                            value = 1 + (diff_far_area - far_diff_min) * far_slope
                            for n in grow_layer:
                                snn.add_spike(frame_idx, n, value, exist='add')
                        elif diff_far_area < -far_diff_min:
                            value = 1 + (-diff_far_area - far_diff_min) * far_slope
                            for n in shrink_layer:
                                snn.add_spike(frame_idx, n, value, exist='add')

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

            if plane_seen and is_blob:
                was_descending = frames_since_descent is not None and frames_since_descent <= recent_descent_frames

            elif plane_missing and was_descending: 
                frames_since_vanish = 0
                was_descending = False

            if frames_since_descent is not None and frames_since_descent < descent_window:
                for n in recent_descent_layer:
                    snn.add_spike(frame_idx, n, 13)

            if frames_since_vanish is not None:
                sec = int(frames_since_vanish // fps)
                if sec < len(vanished_layer):
                    snn.add_spike(frame_idx, vanished_layer[sec], 13)

            if near_y is None:
                old_near_area = None
            if far_y is None:
                old_far_area = None

            near_y_old = near_y
            far_y_old = far_y
            frame_idx += 1

        if frames_plane_missing == 0:
            frame_disappeared = frame_idx - 1

        if frame_first_seen is not None:
            writer.writerow([frame_first_seen, frame_disappeared])

        return frame_idx, video_start
    
frame_idx, _ = run_model(video9)

# Near
descent_near_ids = []
for n in descent_near_layer:                # SuperNeuroMAT looks at neurons as idx, so need ids for each
    descent_near_ids.append(n.idx)

ascent_near_ids = []
for neuron in ascent_near_layer:
    ascent_near_ids.append(neuron.idx)

# Far
descent_far_ids = []
for neuron in descent_far_layer:
    descent_far_ids.append(neuron.idx)

ascent_far_ids = []
for neuron in ascent_far_layer:
    ascent_far_ids.append(neuron.idx)

# Size (grow/shrink)
grow_ids = []
for neuron in grow_layer:
    grow_ids.append(neuron.idx)

shrink_far_ids = []
for neuron in shrink_layer:
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

traffic_torwards_ids = []
for neuron in traffic_torwards_layer:
    traffic_torwards_ids.append(neuron.idx)

traffic_away_ids = []
for neuron in traffic_away_layer:
    traffic_away_ids.append(neuron.idx)

# Decision layers:
takeoff_decision_ids = []
for neuron in takeoff_decision:
    takeoff_decision_ids.append(neuron.idx)

landing_decision_ids = []
for neuron in landing_decision:
    landing_decision_ids.append(neuron.idx)

touch_go_decision_ids = []
for neuron in touch_go_decision:
    touch_go_decision_ids.append(neuron.idx)

away_decision_ids = []
for neuron in traffic_away_decision:
    away_decision_ids.append(neuron.idx)

torwards_decision_ids = []
for neuron in traffic_torwards_decision:
    torwards_decision_ids.append(neuron.idx)

W = snn.weight_mat()

descent_landing_before = W[np.ix_(descent_near_ids[:11], landing_ids)].mean()
descent_takeoff_before = W[np.ix_(descent_near_ids[:11], takeoff_ids)].mean()

ascent_landing_before = W[np.ix_(ascent_far_ids[:11], landing_ids)].mean()
ascent_takeoff_before = W[np.ix_(ascent_far_ids[:11], takeoff_ids)].mean()

vanish_landing_before = W[np.ix_(vanish_ids, landing_ids)].mean()


snn.simulate(time_steps=frame_idx) # simulates full stdp training with teacher spike

print("Vanish spikes:", snn.ispikes[:, vanish_ids].sum())
print("Takeoff teacher:", teacher_count["takeoff"] * 10, "actual:", snn.ispikes[:, takeoff_ids].sum())
print("Landing teachers", teacher_count["landing"] * 10, "actual:", snn.ispikes [:, landing_ids].sum())
print("Touch/go teacher:", teacher_count["touch/go"] * 10, "actual:", snn.ispikes[:, touch_go_ids].sum())

W = snn.weight_mat() # saves the NEW weights after training from training spikes

descent_landing_after = W[np.ix_(descent_near_ids[:11], landing_ids)].mean()
descent_takeoff_after = W[np.ix_(descent_near_ids[:11], takeoff_ids)].mean()

ascent_landing_after = W[np.ix_(ascent_far_ids[:11], landing_ids)].mean()
ascent_takeoff_after = W[np.ix_(ascent_far_ids[:11], takeoff_ids)].mean()

vanish_landing_after = W[np.ix_(vanish_ids, landing_ids)].mean()



print()
print("BEFORE:")
print("Descent-landing:", descent_landing_before)
print("Descent-takeoff:", descent_takeoff_before)
print("Ascent-landing:", ascent_landing_before)
print("Ascent-takeoff:", ascent_takeoff_before)
print("Vanish-landing:", vanish_landing_before)

print()


print("AFTER:")
print("Descent-landing:", descent_landing_after, "percent change:", ((descent_landing_after - descent_landing_before) / descent_landing_before) * 100)
print("Descent-takeoff:", descent_takeoff_after, "percent change:", ((descent_takeoff_after - descent_takeoff_before) / descent_takeoff_before) * 100)
print("Ascent-landing:", ascent_landing_after, "percent change:", ((ascent_landing_after - ascent_landing_before) / ascent_landing_before) * 100)
print("Ascent-takeoff:", ascent_takeoff_after, "percent change:", ((ascent_takeoff_after - ascent_takeoff_before) / ascent_takeoff_before) * 100)
print("Vanish-landing:", vanish_landing_after, "percent change:", (vanish_landing_after - vanish_landing_before) / vanish_landing_before * 100)
print()


### VIDEO TESTING
# video10 = "/Users/aymanaghel/Desktop/LIF/2026_10_08 muted/10muted.mp4"
print("VIDEO TESTING BELOW:")
snn.stdp_setup(Apos, Aneg, positive_update=False, negative_update=False) # turn off stdp training; now in testing
snn.reset()                                                              # to clear spike history for accurate spike counts.

frame_num, video_start = run_model(video9)
snn.simulate(time_steps=frame_num)

print()
print("NEAR Descent layer", snn.ispikes[video_start:(video_start+frame_num), descent_near_ids ].sum())
print("NEAR Ascent layer", snn.ispikes[video_start:(video_start+frame_num), ascent_near_ids ].sum()) # 0; not possible
print("FAR Descent layer", snn.ispikes[video_start:(video_start+frame_num), descent_far_ids ].sum()) # 0; not possible
print("FAR Ascent layer", snn.ispikes[video_start:(video_start+frame_num), ascent_far_ids ].sum())
print("Vanished layer", snn.ispikes[video_start:(video_start+frame_num), vanish_ids ].sum())
print("Takeoff layer", snn.ispikes[video_start:(video_start+frame_num), takeoff_ids ].sum())
print("Landing layer", snn.ispikes[video_start:(video_start+frame_num), landing_ids ].sum())
print("Touch/go layer", snn.ispikes[video_start:(video_start+frame_num), touch_go_ids ].sum())
print()

rows = snn.ispikes[0 : frame_num]

landings    = rows[:, landing_decision[0].idx].sum()
takeoffs    = rows[:, takeoff_decision[0].idx].sum()
touch_gos   = rows[:, touch_go_decision[0].idx].sum()

print("Landings:", landings, " Takeoffs:", takeoffs, " Touch-and-goes:", touch_gos)
print("Operations:", landings + takeoffs + 2 * touch_gos)

print("Landing at frames:", np.where(rows[:, landing_decision[0].idx])[0])
print("Takeoff at frames:", np.where(rows[:, takeoff_decision[0].idx])[0])
print("Touch & go at frames:", np.where(rows[:, touch_go_decision[0].idx])[0])

measure = {
    "landing":  landing_decision_ids,
    "takeoff":  takeoff_decision_ids,
    "touch/go": touch_go_decision_ids,
    "landing_layer": landing_ids,
    "takeoff_layer": takeoff_ids,
    "touch_go_layer": touch_go_ids,
}

labels = ["takeoff", "landing", "touch/go"]
results = {lab: {layer: [] for layer in measure} for lab in labels}

lag = 30   # frames after the event ends to keep counting

with open(video9_labels) as f:
    next(f)
    for row in f:
        row = row.strip()
        if not row:
            continue
        start, end, area, movement, direction = [x.strip() for x in row.split(',')]
        start, end = int(start), int(end)

        window = rows[start:end + lag]
        for layer, ids in measure.items():
            avg_frame = window[:, ids].sum()
            results[movement][layer].append(avg_frame)

for lab in labels:
    print(f"\n{lab.upper()} events:")
    for layer, vals in results[lab].items():
        if not vals:
            continue
        v = np.array(vals)
        print(f"  {layer:9} min {v.min()}  median {np.median(v)}  max {v.max()}  mean {v.mean():.2f}")