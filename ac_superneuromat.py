### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np

rng = np.random.default_rng(0)

snn = SNN()

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

far_return_min_area = 20                                                    # far-box blob must be this big to count as the vanished plane returning
vanish_dead_sec = 8                                                        # first seconds of the vanish timer: a plane showing up now is a different aircraft
plane_gone_sec = 15                                                         # confirmed missing this long = the plane has left (longer than in-event tracking gaps, < vanish_dead_sec)

# dy bands (moved out of run_model): the min is each thermometer's lowest threshold, the max is a veto neuron
min_near_delta = 1.0             # |dy| floor, away and torwards (was min_near_delta_away / min_near_delta_torwards)
max_near_delta_away = 6.0
max_near_delta_torwards = 6.9
min_far_delta = 0.25             # |far dy| floor, away and torwards
max_far_delta = 1.5
far_dy_scale = 4.0               # far |dy| (0.25-1.5, moves in 0.5 px steps) -> near's 1-6 range, so far motion fires the thermometer layers

FPS = 30000 / 1001               # footage is 29.97 fps; every timer below counts frames
SEEN_N = int(2 * FPS) + 1        # was: n_seen > min_frames_plane_seen (2 s)
INH = -1e6                       # bigger than any value injected (even a blob swap), so it always wins
EPS = 1e-6                       # dy sits on a 0.5 px grid; keeps the old strict "<" at the max

def frames(sec):
    return int(sec * FPS)

### Building blocks: each one replaces a kind of python if / counter / boolean
def gate(threshold, **kw):       # leak=inf: forgets every step, fires if this step's input > threshold
    return snn.create_neuron(threshold=threshold, **kw)

def latch():                     # boolean: self-loop holds it on, a -10 turns it off
    n = snn.create_neuron(threshold=0.5)
    snn.create_synapse(n, n, weight=1.0)
    return n

def counter(n):                  # frame counter: +1 per input, 0.5 leak -> fires on the nth input, drains on gaps
    return snn.create_neuron(threshold=0.5 * n + 0.25, leak=0.5)

def connect(pres, post, w=1.0):
    for p in pres:
        snn.create_synapse(p, post, weight=w)

def set_reset(l, on, off):       # latch = True on "on", False on "off"
    snn.create_synapse(on, l, weight=1.0)
    snn.create_synapse(off, l, weight=-10.0)

def edge(src):                   # was: seen and not was_seen -> fires on src's first step only
    e, relay = gate(0.5), gate(0.5)          # relay: SuperNeuroMAT allows one synapse per pre/post pair
    snn.create_synapse(src, e, weight=1.0)
    snn.create_synapse(src, relay, weight=1.0)
    snn.create_synapse(relay, e, weight=-1.0)
    return e

def one_shot(src, rearm, n):     # was: n_confirmed == plane_gone_frames -> fires once n steps into src, silent until rearm
    c, done = counter(n), latch()
    snn.create_synapse(src, c, weight=1.0)
    snn.create_synapse(done, c, weight=-1.0)
    set_reset(done, c, rearm)
    return c

def presence():                  # was: n_seen / n_missing counters + seen / missing booleans
    present_count, missing_count = counter(SEEN_N), counter(SEEN_N)
    seen, missing = latch(), latch()
    set_reset(seen, present_count, missing_count)
    set_reset(missing, missing_count, present_count)
    return present_count, missing_count, seen, missing

# presence: encoder sends raw "blob this frame" facts, debounce lives here
plane_present_count, plane_missing_count, plane_seen, plane_absent = presence()   # plane_seen = a blob in either box for 2 s, plane_absent = no blob for 2 s
near_present_count, near_missing_count, near_seen, near_missing = presence()         # was: box["near"]
far_present_count, far_missing_count, far_seen, far_missing = presence()             # was: box["far"]

blob_detected_layer = [gate(0.5)]
connect([plane_seen], blob_detected_layer[0])                              # blob_detected fires while plane_seen is on

# 25 neurons each - eyes of SNN (input layers)
# thermometers: lowest threshold = band min, veto = band max, plane_absent blocks them until plane_seen
def thermometer(floor):
    return [gate(max((i+1) * 0.5, floor)) for i in range(25)]

descent_near_layer = thermometer(min_near_delta)
ascent_near_layer = thermometer(min_near_delta)

descent_far_layer = thermometer(min_far_delta * far_dy_scale)
ascent_far_layer = thermometer(min_far_delta * far_dy_scale)

thermometers = {"descent_near": descent_near_layer, "ascent_near": ascent_near_layer,
                "descent_far": descent_far_layer, "ascent_far": ascent_far_layer}
veto = {"descent_near": gate(max_near_delta_away - EPS),
        "ascent_near": gate(max_near_delta_torwards - EPS),
        "descent_far": gate(max_far_delta * far_dy_scale - EPS),
        "ascent_far": gate(max_far_delta * far_dy_scale - EPS)}
for name, layer in thermometers.items():
    for n in layer:
        snn.create_synapse(veto[name], n, weight=INH)
        snn.create_synapse(plane_absent, n, weight=INH)

grow_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
shrink_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
for n in grow_layer + shrink_layer:
    snn.create_synapse(plane_absent, n, weight=INH)                         # plane_absent blocks grow/shrink until plane_seen

# recent descent: was descent_streak >= min_descent_frames + frames_since_descent <= recent_descent_frames (10 s)
descent_count = counter(5)
recent_descent = latch()
recent_timer = counter(frames(10))
snn.create_synapse(descent_near_layer[0], descent_count, weight=1.0)
set_reset(recent_descent, descent_count, recent_timer)
snn.create_synapse(recent_descent, recent_timer, weight=1.0)
snn.create_synapse(descent_near_layer[0], recent_timer, weight=-1.0)       # still descending = timer drains, so 10 s counts from the end
recent_descent_layer = [recent_descent]                                     # one neuron because recent_descent is yes/no

# was_descending + vanish (was: near seen + recent -> was_descending, near missing + was_descending -> frames_since_vanish = 0)
was_descending = latch()
descent_and_seen = gate(1.5)
connect([near_seen, recent_descent], descent_and_seen)
vanish_trigger = gate(1.5, refractory_period=5)                             # refractory = fires once
connect([near_missing, was_descending], vanish_trigger)
set_reset(was_descending, descent_and_seen, vanish_trigger)

# vanish timer: 45 latches that hand off to the next one every second (was: frames_since_vanish // fps)
vanished_layer = [latch() for i in range(45)]                               # 45 neurons to make the layer into a timer, one neuron / sec
for k, cell in enumerate(vanished_layer):
    tick = counter(round(FPS) - 1)                                          # 30-frame cells
    snn.create_synapse(cell, tick, weight=1.0)
    snn.create_synapse(tick, cell, weight=-10.0)
    if k + 1 < len(vanished_layer):
        snn.create_synapse(tick, vanished_layer[k + 1], weight=1.0)
snn.create_synapse(vanish_trigger, vanished_layer[0], weight=1.0)
for cell in vanished_layer[1:]:
    snn.create_synapse(vanish_trigger, cell, weight=-10.0)                 # new vanish restarts the timer
window_open = gate(0.5)                                                     # was: vanish_dead_zone <= frames_since_vanish < vanish_frames
connect(vanished_layer[vanish_dead_sec:], window_open)

# returned: a box's first seen frame inside the window; far blob must be plane-sized (was: median(areas) >= far_return_min_area)
far_big = gate(far_return_min_area - 0.5)
far_sized_count, far_sized = counter(SEEN_N), latch()
snn.create_synapse(far_big, far_sized_count, weight=1.0)
set_reset(far_sized, far_sized_count, far_missing_count)
near_return, far_return = gate(1.5), gate(1.5)
connect([edge(near_seen), window_open], near_return)
connect([edge(far_sized), window_open], far_return)
for r in (near_return, far_return):
    for cell in vanished_layer:
        snn.create_synapse(r, cell, weight=-10.0)                          # was: frames_since_vanish = None

# gone pulses: once per empty stretch (was: n_confirmed / frames_confirmed_missing == plane_gone_frames)
plane_gone_layer = [one_shot(plane_absent, plane_present_count, frames(plane_gone_sec))]   # one-frame pulse when BOTH boxes have been empty 15 s
near_gone_layer = [one_shot(near_missing, near_present_count, frames(plane_gone_sec))]     # one-frame pulse when the near box has been empty 15 s
far_gone_layer = [one_shot(far_missing, far_present_count, frames(plane_gone_sec))]        # one-frame pulse when the far box has been empty 15 s

# returned latch per box (was: returned + returned_box), cleared by that box's gone pulse or after 75 s
returned_near, returned_far, returned_timer = latch(), latch(), counter(frames(75))
set_reset(returned_near, near_return, near_gone_layer[0])
set_reset(returned_far, far_return, far_gone_layer[0])
for r in (returned_near, returned_far):
    snn.create_synapse(r, returned_timer, weight=1.0)
    snn.create_synapse(returned_timer, r, weight=-10.0)
returned_layer = [gate(0.5)]                                                # on while a vanished plane is back (came back after the dead zone)
connect([returned_near, returned_far], returned_layer[0])

learning_inputs = descent_near_layer + ascent_near_layer +\
         descent_far_layer + ascent_far_layer

sizing_inputs = grow_layer + shrink_layer

memory_inputs = recent_descent_layer + vanished_layer + returned_layer + plane_gone_layer + near_gone_layer + far_gone_layer + blob_detected_layer

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
# Event decisions: accumulator sums evidence over seconds -> decision fire once -> gate (which locks until the plane leaves)
landing_accumulator = [snn.create_neuron(threshold=5, leak=0.05, reset_state=0)]
takeoff_accumulator = [snn.create_neuron(threshold=8, leak=0.025, reset_state=0)]
touch_go_accumulator = [snn.create_neuron(threshold=10, leak=0.05, reset_state=0)]

landing_decision = [snn.create_neuron(threshold=0.5, refractory_period=2)]     # leak=inf: fires on accumulator AND NOT gate
takeoff_decision = [snn.create_neuron(threshold=0.5, refractory_period=2)]
touch_go_decision = [snn.create_neuron(threshold=0.5, refractory_period=2)]

landing_gate = [snn.create_neuron(threshold=0.5)]                               # latch: on after deciding, off on plane_gone
takeoff_gate = [snn.create_neuron(threshold=0.5)]
touch_go_gate = [snn.create_neuron(threshold=0.5)]

traffic_away_decision = [snn.create_neuron(threshold=1)]
traffic_torwards_decision = [snn.create_neuron(threshold=1)]

aircraft_detected_decision = [snn.create_neuron(threshold=1)]

decision_neurons = landing_decision + takeoff_decision + touch_go_decision + traffic_away_decision + traffic_torwards_decision

# each gate unlocks when the box its event happens in empties: landing = near, takeoff = far, touch/go = both boxes
event_decisions = [(landing_layer, landing_accumulator, landing_decision, landing_gate, near_gone_layer),
                   (takeoff_layer, takeoff_accumulator, takeoff_decision, takeoff_gate, far_gone_layer),
                   (touch_go_layer, touch_go_accumulator, touch_go_decision, touch_go_gate, plane_gone_layer)]

for layer, acc, decision, gate, gone in event_decisions:
    for pre in layer:
        snn.create_synapse(pre, acc[0], weight=0.1, stdp_enabled=False)        # one full 10-neuron burst = 1.0
    snn.create_synapse(acc[0], decision[0], weight=1.0, stdp_enabled=False)
    snn.create_synapse(decision[0], gate[0], weight=1.0, stdp_enabled=False)    # deciding locks the gate
    snn.create_synapse(gate[0], gate[0], weight=1.0, stdp_enabled=False)        # self-loop holds it
    snn.create_synapse(gate[0], decision[0], weight=-2.0, stdp_enabled=False)   # locked gate blocks the decision
    snn.create_synapse(gone[0], gate[0], weight=-5.0, stdp_enabled=False)           # plane left? unlock
    snn.create_synapse(gone[0], acc[0], weight=-0.5 * acc[0].threshold, stdp_enabled=False)  # clear leftover evidence

output_decisions = [(traffic_away_layer, traffic_away_decision),
                    (traffic_torwards_layer, traffic_torwards_decision),
                    (aircraft_detected_layer, aircraft_detected_decision)]

for layer, decision in output_decisions:
    for pre in layer:
        for post in decision:
            snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

for pre in learning_inputs:
    for post in landing_layer + takeoff_layer:          # touch/go is decided by returned; learned direction weights only added noise
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

for pre in sizing_inputs:
    for post in sizing_outputs:
        snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

for pre in vanished_layer:
    for post in landing_layer:
        snn.create_synapse(pre, post, weight=rng.uniform(0.0035, 0.0084), stdp_enabled=True)

for pre in blob_detected_layer:
    for post in aircraft_detected_layer:
        snn.create_synapse(pre, post, weight=1.0, stdp_enabled=False)

# Feature neurons: fixed logic (no STDP, leak=inf so each is a same-frame detector)
# takeoff  = ascent AND NOT returned
# touch/go = returned (came down, left, came back within the window); fires on returned alone
takeoff_feat = snn.create_neuron(threshold=0.5)
touch_go_feat = snn.create_neuron(threshold=1.0)

ascent_any = [ascent_near_layer[0], ascent_far_layer[0]]     # any ascent this frame
for pre in ascent_any:
    snn.create_synapse(pre, takeoff_feat, weight=0.6, delay=5, stdp_enabled=False)    # 1 ascent alone fires takeoff_feat; delay 5 = returned's circuit settles first
    snn.create_synapse(pre, touch_go_feat, weight=0.4, stdp_enabled=False)   # 2 ascents = 0.8, not enough alone

for pre in returned_layer:
    snn.create_synapse(pre, takeoff_feat, weight=-2.0, stdp_enabled=False)
    snn.create_synapse(pre, touch_go_feat, weight=1.1, stdp_enabled=False)    # back after vanishing = touch/go evidence on its own

feature_weight = 0.6        # feature -> output weight used in testing
feature_synapses = []       # kept at 0 during training (teacher-only learning), set to feature_weight after
for feat, layer in [(takeoff_feat, takeoff_layer), (touch_go_feat, touch_go_layer)]:
    for post in layer:
        feature_synapses.append(snn.create_synapse(feat, post, weight=0.0, stdp_enabled=False))

# landing = still gone late in the vanish timer (no ascent)
landing_sec = 39            # landing_feat fires once the timer passes this (longest touch/go gap in video 9 was 34 s)
ramp_tau = 6.0              # seconds; how sharply rises
landing_feat = snn.create_neuron(threshold=1.0)

for k, pre in enumerate(vanished_layer):
    if k >= vanish_dead_sec:
        w = np.exp((k - landing_sec) / ramp_tau)                            # 0.04 at 20 s, 1.0 at 39 s, 1.18 at 40 s
        snn.create_synapse(pre, landing_feat, weight=w, stdp_enabled=False)
for pre in ascent_any:
    snn.create_synapse(pre, landing_feat, weight=-2.0, stdp_enabled=False)  # climbing = not a landing

for post in landing_layer:
    feature_synapses.append(snn.create_synapse(landing_feat, post, weight=0.0, stdp_enabled=False))

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

defined_outputs = {"takeoff": takeoff_layer,
                   "landing": landing_layer,
                   "touch/go": touch_go_layer}

defined_directions = {"away": traffic_away_layer, 
                      "torwards": traffic_torwards_layer}

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses; not used
snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

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

def run_reverse(video, chunk=100):
    cap = cv2.VideoCapture(video)
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while end > 0:
        start = max(0, end-chunk)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end-start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        yield from reversed(frames)
        end = start
    cap.release()

def run_forward(video):
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        yield frame

def run_model(video, reverse_video):
    source = run_reverse(video) if reverse_video else run_forward(video)
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

    snn.add_spike(0, plane_absent, 1.0)                                    # plane_absent starts on, so plane_seen starts off

    for frame in source:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = backSub.apply(gray)
        near_mask = mask.copy()
        near_mask[604:755, 974:1239] = 0
        near_y, near_area = track(near_mask, NEAR_BOX, min_near_area, max_near_area)
        far_y, far_area = track(mask, FAR_BOX, min_far_area, max_far_area)

        # presence: raw "blob this frame" facts in, the presence() circuits do the debouncing
        is_blob = float(near_y is not None or far_y is not None)
        snn.add_spike(frame_idx, plane_present_count, is_blob)
        snn.add_spike(frame_idx, plane_missing_count, 1.0 - is_blob)
        for present_count, missing_count, y in ((near_present_count, near_missing_count, near_y),
                                                (far_present_count, far_missing_count, far_y)):
            present = float(y is not None)
            snn.add_spike(frame_idx, present_count, present)
            snn.add_spike(frame_idx, missing_count, 1.0 - present)
        if far_y is not None:
            snn.add_spike(frame_idx, far_big, far_area)                     # far_big checks plane-sized

        # motion: signed value in, thermometer floors + vetoes do the band checks, plane_absent blocks it until plane_seen
        for y, y_old, scale, descent, ascent in ((near_y, near_y_old, 1.0, "descent_near", "ascent_near"),
                                                 (far_y, far_y_old, far_dy_scale, "descent_far", "ascent_far")):
            if y is not None and y_old is not None:
                value = (y - y_old) * scale                                 # = -dy * scale: positive when moving down the screen
                snn.add_spike(frame_idx, veto[descent], value)
                snn.add_spike(frame_idx, veto[ascent], -value)
                for n in thermometers[descent]:
                    snn.add_spike(frame_idx + 1, n, value)                  # +1 frame so it lands with its veto
                for n in thermometers[ascent]:
                    snn.add_spike(frame_idx + 1, n, -value)

        # grow / shrink (left as python for now)
        if near_y is not None and near_y_old is not None:
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

        if near_y is None:
            old_near_area = None
        if far_y is None:
            old_far_area = None

        near_y_old = near_y
        far_y_old = far_y
        frame_idx += 1

    return frame_idx
    
frame_idx = run_model(video9, reverse_video=False)

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

for syn in feature_synapses:                     # training done: switch the feature neurons on for testing
    syn.weight = feature_weight

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
print("VIDEO TESTING BELOW:")
snn.stdp_setup(Apos, Aneg, positive_update=False, negative_update=False) # turn off stdp training; now in testing
snn.reset()                                                              # to clear spike history for accurate spike counts.

need_diagnostic = False
if need_diagnostic == True:

    frame_num = run_model(video10, reverse_video=False)
    snn.simulate(time_steps=frame_num)
    print()
    print("NEAR Descent layer", snn.ispikes[0:frame_num, descent_near_ids ].sum())
    print("NEAR Ascent layer", snn.ispikes[0:frame_num, ascent_near_ids ].sum())
    print("FAR Descent layer", snn.ispikes[0:frame_num, descent_far_ids ].sum())
    print("FAR Ascent layer", snn.ispikes[0:frame_num, ascent_far_ids ].sum())
    print("Vanished layer", snn.ispikes[0:frame_num, vanish_ids ].sum())
    print("Takeoff layer", snn.ispikes[0:frame_num, takeoff_ids ].sum())
    print("Landing layer", snn.ispikes[0:frame_num, landing_ids ].sum())
    print("Touch/go layer", snn.ispikes[0:frame_num, touch_go_ids ].sum())
    print()

def test_video(video, video_label, diagnostic_enabled):
    snn.stdp_setup(Apos, Aneg, positive_update=False, negative_update=False)
    snn.reset()

    frame_num = run_model(video, reverse_video=True)
    snn.simulate(time_steps=frame_num)
    rows = snn.ispikes[0 : frame_num]

    landings    = rows[:, landing_decision[0].idx].sum()
    takeoffs    = rows[:, takeoff_decision[0].idx].sum()
    touch_gos   = rows[:, touch_go_decision[0].idx].sum()

    print("Landings:", landings, " Takeoffs:", takeoffs, " Touch-and-goes:", touch_gos)
    print("Operations:", landings + takeoffs + 2 * touch_gos) # In official reporting, touch/go counts as 2 movements
    print()

    if diagnostic_enabled == True:
        measure = {
            "landing":  landing_decision_ids,
            "takeoff":  takeoff_decision_ids,
            "touch/go": touch_go_decision_ids,
            "landing_layer": landing_ids,
            "takeoff_layer": takeoff_ids,
            "touch_go_layer": touch_go_ids,
        }


        print("Landing at frames:", np.where(rows[:, landing_decision[0].idx])[0])
        print("Takeoff at frames:", np.where(rows[:, takeoff_decision[0].idx])[0])
        print("Touch & go at frames:", np.where(rows[:, touch_go_decision[0].idx])[0])

        labels = ["takeoff", "landing", "touch/go"]
        results = {lab: {layer: [] for layer in measure} for lab in labels}

        lag = {"takeoff": 5 * 30, "landing": 45 * 30, "touch/go": 5 * 30}   # frames after the event ends to keep counting (around 30 fps); landing decides around 40 sec after the plane leaves
        lead = {"takeoff": 10 * 30, "landing": 0, "touch/go": 0} # for accurate scoring 

        with open(video_label) as f:
            next(f)
            for row in f:
                row = row.strip()
                if not row:
                    continue
                start, end, area, movement, direction = [x.strip() for x in row.split(',')]
                start, end = int(start), int(end)

                window = rows[start - lead[movement]:end + lag[movement]]
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

print("VIDEO10 test")
test_video(video10, video10_labels, diagnostic_enabled=True)

unlabeled_videos = [video1, video2, video3, video4, video5, video6, video7, video8]

print("Unlabeled tests")
idx = 0
for video in unlabeled_videos:
    print(f"Video {idx+1}")
    test_video(video, None, diagnostic_enabled=False)
    idx += 1