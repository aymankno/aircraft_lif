from superneuromat import SNN
import numpy as np
from types import SimpleNamespace

rng = np.random.default_rng(0)

landing_sec = 39
ramp_tau = 6.0
feature_weight = 0.6
plane_gone_sec = 15
vanish_dead_sec = 8
far_return_min_area = 20

NEAR_BOX = (468, 1, 967, 747)
FAR_BOX  = (974, 604, 265, 151)

def build_network():
    snn = SNN()

    # 25 neurons each - eyes of SNN (input layers)
    descent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
    ascent_near_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

    descent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
    ascent_far_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

    grow_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
    shrink_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

    recent_descent_layer = [snn.create_neuron(threshold=0.5)]                   # one neuron because recent_descent is yes/no
    vanished_layer = [snn.create_neuron(threshold=(0.5)) for i in range(45)]    # 45 neurons to make the layer into a timer, one neuron / sec
    returned_layer = [snn.create_neuron(threshold=0.5)]                         # on while a vanished plane is back (came back after the dead zone)
    plane_gone_layer = [snn.create_neuron(threshold=0.5)]                       # one-frame pulse when BOTH boxes have been empty 15 s
    near_gone_layer = [snn.create_neuron(threshold=0.5)]                        # one-frame pulse when the near box has been empty 15 s
    far_gone_layer = [snn.create_neuron(threshold=0.5)]                         # one-frame pulse when the far box has been empty 15 s

    blob_detected_layer = [snn.create_neuron(threshold=0.5)]

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
        snn.create_synapse(pre, takeoff_feat, weight=0.6, stdp_enabled=False)    # 1 ascent alone fires takeoff_feat
        snn.create_synapse(pre, touch_go_feat, weight=0.4, stdp_enabled=False)   # 2 ascents = 0.8, not enough alone

    for pre in returned_layer:
        snn.create_synapse(pre, takeoff_feat, weight=-2.0, stdp_enabled=False)
        snn.create_synapse(pre, touch_go_feat, weight=1.1, stdp_enabled=False)    # back after vanishing = touch/go evidence on its own

    feature_synapses = []       # kept at 0 during training (teacher-only learning), set to feature_weight after
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

        return SimpleNamespace(
        snn=snn,
        # inputs
        descent_near_layer=descent_near_layer, ascent_near_layer=ascent_near_layer,
        descent_far_layer=descent_far_layer, ascent_far_layer=ascent_far_layer,
        grow_layer=grow_layer, shrink_layer=shrink_layer,
        # memory
        recent_descent_layer=recent_descent_layer, vanished_layer=vanished_layer,
        returned_layer=returned_layer, plane_gone_layer=plane_gone_layer,
        near_gone_layer=near_gone_layer, far_gone_layer=far_gone_layer,
        blob_detected_layer=blob_detected_layer,
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
        feature_synapses=feature_synapses,
        # ids
        descent_near_ids=descent_near_ids, ascent_near_ids=ascent_near_ids,
        descent_far_ids=descent_far_ids, ascent_far_ids=ascent_far_ids,
        grow_ids=grow_ids, shrink_far_ids=shrink_far_ids, vanish_ids=vanish_ids,
        landing_ids=landing_ids, takeoff_ids=takeoff_ids, touch_go_ids=touch_go_ids,
        traffic_torwards_ids=traffic_torwards_ids, traffic_away_ids=traffic_away_ids,
        takeoff_decision_ids=takeoff_decision_ids, landing_decision_ids=landing_decision_ids,
        touch_go_decision_ids=touch_go_decision_ids,
        away_decision_ids=away_decision_ids, torwards_decision_ids=torwards_decision_ids,
    )