from superneuromat import SNN
import numpy as np
from types import SimpleNamespace

rng = np.random.default_rng(0)

FPS = 30000 / 1001               # footage is 29.97 fps; every timer below counts frames
SEEN_N = int(2 * FPS) + 1        # was: n_seen > min_frames_plane_seen (2 s)
INH = -1e6                       # bigger than any value injected (even a blob swap), so it always wins
EPS = 1e-6                       # dy sits on a 0.5 px grid; keeps the old strict "<" at the max

NEAR_BOX = (468, 1, 967, 747)
FAR_BOX  = (974, 604, 265, 151)

landing_sec = 39            # landing_feat fires once the timer passes this (longest touch/go gap in video 9 was 34 s)
ramp_tau = 6.0              # seconds; how sharply rises
feature_weight = 0.6
far_return_min_area = 20
vanish_dead_sec = 8
plane_gone_sec = 15
min_near_delta = 1.0
max_near_delta_away = 6.0
max_near_delta_torwards = 6.9
min_far_delta = 0.25
max_far_delta = 1.5
far_dy_scale = 4.0
near_climb_frames = 30      # frames of near ascent (not returned) to call a takeoff; tune on reversed video 9

def build_network():
    snn = SNN()

    def frames(sec):
        return int(sec * FPS)

    def gate(threshold, **kw):       # leak=inf: forgets every step, fires if this step's input > threshold
        return snn.create_neuron(threshold=threshold, **kw)

    def latch():                     # boolean: self-loop holds it on, a -10 turns off
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
    down_steps = [gate(0.25), gate(0.75)]
    up_steps = [gate(0.25), gate(0.75)]
    slow_win = gate(2.75)
    slow_rise = gate(2.75)
    for steps, w in ((down_steps, 0.5), (up_steps, -0.5)):
        for s in steps:
            prev = s
            for k in range(15):
                r = gate(0.5)
                snn.create_synapse(prev, r, weight=1.0)
                snn.create_synapse(r, slow_win, weight=w)
                snn.create_synapse(r, slow_rise, weight=-w)
                prev = r
    slow_desc_acc = snn.create_neuron(threshold=30, leak=0.3)
    snn.create_synapse(slow_win, slow_desc_acc, weight=1.0)
    slow_ok = gate(0.5)
    snn.create_synapse(slow_desc_acc, slow_ok, weight=1.0)
    snn.create_synapse(slow_ok, recent_descent, weight=1.0)
    rise_acc = snn.create_neuron(threshold=30, leak=0.3)
    snn.create_synapse(slow_rise, rise_acc, weight=1.0)
    near_rising = latch()
    set_reset(near_rising, rise_acc, near_missing_count)
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

    # far-box mirror for torwards traffic: descend in far, disappear from far -> same vanish timer. For reverse traffic
    far_descent_count = snn.create_neuron(threshold=5, leak=0.1)
    far_recent_descent = latch()
    far_recent_timer = counter(frames(10))
    snn.create_synapse(descent_far_layer[0], far_descent_count, weight=1.0)
    snn.create_synapse(ascent_far_layer[0], far_descent_count, weight=-1.0)
    set_reset(far_recent_descent, far_descent_count, far_recent_timer)
    snn.create_synapse(far_recent_descent, far_recent_timer, weight=1.0)
    snn.create_synapse(descent_far_layer[0], far_recent_timer, weight=-1.0)
    far_was_descending = latch()
    far_descent_and_seen = gate(1.5)
    connect([far_seen, far_recent_descent], far_descent_and_seen)
    far_vanish_trigger = gate(1.5, refractory_period=5)
    connect([far_missing, far_was_descending], far_vanish_trigger)
    set_reset(far_was_descending, far_descent_and_seen, far_vanish_trigger)

    # vanish timer: 45 latches that hand off to the next one every second (was: frames_since_vanish // fps)
    vanished_layer = [latch() for i in range(45)]                               # 45 neurons to make the layer into a timer, one neuron / sec
    for k, cell in enumerate(vanished_layer):
        tick = counter(round(FPS) - 1)                                          # 30-frame cells
        snn.create_synapse(cell, tick, weight=1.0)
        snn.create_synapse(tick, cell, weight=-10.0)
        if k + 1 < len(vanished_layer):
            snn.create_synapse(tick, vanished_layer[k + 1], weight=1.0)
    for trigger in (vanish_trigger, far_vanish_trigger):                       # either box's vanish starts the timer
        snn.create_synapse(trigger, vanished_layer[0], weight=1.0)
        for cell in vanished_layer[1:]:
            snn.create_synapse(trigger, cell, weight=-10.0)                    # new vanish restarts the timer
    window_open = gate(0.5)                                                     # was: vanish_dead_zone <= frames_since_vanish < vanish_frames
    connect(vanished_layer[vanish_dead_sec:], window_open)
    for cell in vanished_layer:
        snn.create_synapse(cell, slow_ok, weight=-1.0)

    # returned: a box's first seen frame inside the window; far blob must be plane-sized (was: median(areas) >= far_return_min_area)
    far_big = gate(far_return_min_area - 0.5)
    far_sized_count, far_sized = counter(SEEN_N), latch()
    snn.create_synapse(far_big, far_sized_count, weight=1.0)
    set_reset(far_sized, far_sized_count, far_missing_count)
    near_back_trig = gate(1.5)
    connect([edge(near_seen), window_open], near_back_trig)
    near_back = latch()
    set_reset(near_back, near_back_trig, near_missing_count)
    near_still = gate(2.5)
    connect([near_back, near_seen, near_rising], near_still)
    stay_acc = snn.create_neuron(threshold=84, leak=0.3)
    snn.create_synapse(near_still, stay_acc, weight=1.0)
    near_stayed = latch()
    set_reset(near_stayed, stay_acc, near_missing_count)
    near_return = edge(near_stayed)
    far_return = gate(1.5)
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

    # each gate unlocks when both boxes have been empty 15 s (traffic can go either direction, so no box owns an event)
    event_decisions = [(landing_layer, landing_accumulator, landing_decision, landing_gate, plane_gone_layer),
                    (takeoff_layer, takeoff_accumulator, takeoff_decision, takeoff_gate, plane_gone_layer),
                    (touch_go_layer, touch_go_accumulator, touch_go_decision, touch_go_gate, plane_gone_layer)]

    for layer, acc, decision, lock, gone in event_decisions:
        for pre in layer:
            snn.create_synapse(pre, acc[0], weight=0.1, stdp_enabled=False)        # one full 10-neuron burst = 1.0
        snn.create_synapse(acc[0], decision[0], weight=1.0, stdp_enabled=False)
        snn.create_synapse(decision[0], lock[0], weight=1.0, stdp_enabled=False)    # deciding locks the gate
        snn.create_synapse(lock[0], lock[0], weight=1.0, stdp_enabled=False)        # self-loop holds it
        snn.create_synapse(lock[0], decision[0], weight=-2.0, stdp_enabled=False)   # locked gate blocks the decision
        snn.create_synapse(gone[0], lock[0], weight=-5.0, stdp_enabled=False)           # plane left? unlock
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

    # near climb (torwards takeoff): short climbs through the near box get their own accumulator, so far climb-outs keep threshold 8
    near_takeoff_feat = snn.create_neuron(threshold=0.5)
    snn.create_synapse(near_back, near_takeoff_feat, weight=-2.0)
    snn.create_synapse(near_back, takeoff_feat, weight=-2.0)
    snn.create_synapse(ascent_near_layer[0], near_takeoff_feat, weight=1.0, delay=5, stdp_enabled=False)   # delay 5 = returned settles first
    for pre in returned_layer:
        snn.create_synapse(pre, near_takeoff_feat, weight=-2.0, stdp_enabled=False)                        # climbing after a return = touch/go
    near_climb_accumulator = snn.create_neuron(threshold=20, leak=0.1)
    snn.create_synapse(near_takeoff_feat, near_climb_accumulator, weight=1.0, stdp_enabled=False)
    snn.create_synapse(near_climb_accumulator, takeoff_decision[0], weight=1.0, stdp_enabled=False)          # takeoff_gate still blocks repeats

    feature_synapses = []
    for feat, layer in [(takeoff_feat, takeoff_layer), (touch_go_feat, touch_go_layer)]:
        for post in layer:
            feature_synapses.append(snn.create_synapse(feat, post, weight=0.0, stdp_enabled=False))

    # landing = still gone late in the vanish timer (no ascent)
    landing_feat = snn.create_neuron(threshold=1.0)

    for k, pre in enumerate(vanished_layer):
        if k >= vanish_dead_sec:
            w = np.exp((k - landing_sec) / ramp_tau)                            # 0.04 at 20 s, 1.0 at 39 s, 1.18 at 40 s
            snn.create_synapse(pre, landing_feat, weight=w, stdp_enabled=False)
    for pre in ascent_any:
        snn.create_synapse(pre, landing_feat, weight=-2.0, stdp_enabled=False)  # climbing = not a landing

    for post in landing_layer:
        feature_synapses.append(snn.create_synapse(landing_feat, post, weight=0.0, stdp_enabled=False))
    landing_clear = gate(0.5)
    snn.create_synapse(plane_gone_layer[0], landing_accumulator[0], weight=0.0, stdp_enabled=False, exist="overwrite")
    snn.create_synapse(plane_gone_layer[0], landing_clear, weight=1.0)
    for cell in vanished_layer:
        snn.create_synapse(cell, landing_clear, weight=-1.0)
    snn.create_synapse(landing_clear, landing_accumulator[0], weight=-0.5 * landing_accumulator[0].threshold)

    # decisions
    takeoff_decision_ids = []
    for neuron in takeoff_decision:
        takeoff_decision_ids.append(neuron.idx)

    landing_decision_ids = []
    for neuron in landing_decision:
        landing_decision_ids.append(neuron.idx)

    touch_go_decision_ids = []
    for neuron in touch_go_decision:
        touch_go_decision_ids.append(neuron.idx)

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


    return SimpleNamespace(
        snn=snn,
        # presence
        plane_present_count=plane_present_count, plane_missing_count=plane_missing_count,
        plane_seen=plane_seen, plane_absent=plane_absent,
        near_present_count=near_present_count, near_missing_count=near_missing_count,
        near_seen=near_seen, near_missing=near_missing,
        far_present_count=far_present_count, far_missing_count=far_missing_count,
        far_seen=far_seen, far_missing=far_missing,
        # inputs
        descent_near_layer=descent_near_layer, ascent_near_layer=ascent_near_layer,
        descent_far_layer=descent_far_layer, ascent_far_layer=ascent_far_layer,
        thermometers=thermometers, veto=veto,
        grow_layer=grow_layer, shrink_layer=shrink_layer,
        # recent descent / slow descent / rising
        descent_count=descent_count, recent_descent=recent_descent, recent_timer=recent_timer,
        down_steps=down_steps, up_steps=up_steps, slow_win=slow_win, slow_rise=slow_rise,
        slow_desc_acc=slow_desc_acc, slow_ok=slow_ok, rise_acc=rise_acc, near_rising=near_rising,
        # vanish (near)
        was_descending=was_descending, descent_and_seen=descent_and_seen, vanish_trigger=vanish_trigger,
        # vanish (far)
        far_descent_count=far_descent_count, far_recent_descent=far_recent_descent,
        far_recent_timer=far_recent_timer, far_was_descending=far_was_descending,
        far_descent_and_seen=far_descent_and_seen, far_vanish_trigger=far_vanish_trigger,
        # vanish timer
        window_open=window_open,
        # returned
        far_big=far_big, far_sized_count=far_sized_count, far_sized=far_sized,
        near_back_trig=near_back_trig, near_back=near_back, near_still=near_still,
        stay_acc=stay_acc, near_stayed=near_stayed, near_return=near_return, far_return=far_return,
        returned_near=returned_near, returned_far=returned_far, returned_timer=returned_timer,
        # memory
        recent_descent_layer=recent_descent_layer, vanished_layer=vanished_layer,
        returned_layer=returned_layer, plane_gone_layer=plane_gone_layer,
        near_gone_layer=near_gone_layer, far_gone_layer=far_gone_layer,
        blob_detected_layer=blob_detected_layer,
        # grouped lists
        learning_inputs=learning_inputs, sizing_inputs=sizing_inputs, memory_inputs=memory_inputs,
        event_outputs=event_outputs, sizing_outputs=sizing_outputs,
        decision_neurons=decision_neurons, event_decisions=event_decisions,
        output_decisions=output_decisions,
        # outputs
        landing_layer=landing_layer, takeoff_layer=takeoff_layer, touch_go_layer=touch_go_layer,
        traffic_away_layer=traffic_away_layer, traffic_torwards_layer=traffic_torwards_layer,
        aircraft_detected_layer=aircraft_detected_layer,
        # decisions
        landing_accumulator=landing_accumulator, takeoff_accumulator=takeoff_accumulator,
        touch_go_accumulator=touch_go_accumulator,
        landing_decision=landing_decision, takeoff_decision=takeoff_decision,
        touch_go_decision=touch_go_decision,
        landing_gate=landing_gate, takeoff_gate=takeoff_gate, touch_go_gate=touch_go_gate,
        traffic_away_decision=traffic_away_decision,
        traffic_torwards_decision=traffic_torwards_decision,
        aircraft_detected_decision=aircraft_detected_decision,
        # features
        takeoff_feat=takeoff_feat, touch_go_feat=touch_go_feat, landing_feat=landing_feat,
        ascent_any=ascent_any,
        near_takeoff_feat=near_takeoff_feat, near_climb_accumulator=near_climb_accumulator,
        landing_clear=landing_clear,
        feature_synapses=feature_synapses,
        # ids
        takeoff_decision_ids=takeoff_decision_ids, landing_decision_ids=landing_decision_ids,
        touch_go_decision_ids=touch_go_decision_ids, descent_near_ids=descent_near_ids, 
        ascent_near_ids=ascent_near_ids, descent_far_ids=descent_far_ids, ascent_far_ids=ascent_far_ids,
        grow_ids=grow_ids, shrink_far_ids=shrink_far_ids, vanish_ids=vanish_ids,
        landing_ids=landing_ids, takeoff_ids=takeoff_ids, touch_go_ids=touch_go_ids,
        traffic_torwards_ids=traffic_torwards_ids, traffic_away_ids=traffic_away_ids,
    )