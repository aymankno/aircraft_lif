### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np

rng = np.random.default_rng(0)

snn = SNN()

# 25 neurons each - eyes of SNN (input layers)
descent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

inputs = descent_layer + ascent_layer

# 10 neurons each -  decision layers (output layers)
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

outputs = landing_layer + takeoff_layer + touch_go_layer

for pre in inputs:                                  # Makes 1500 synapses to connect everything together
    for post in outputs:                            # Weight is anywhere in range with avg of .25, 1/4 of 1.0 (thresh)
        snn.create_synapse(pre, post, weight=rng.uniform(0.1, 0.4), stdp_enabled=True)


start_takeoff = 1500 # starting frame; not real, for testing
end_takeoff = 2000   # ending frame; not real, for testing

start_landing = 4500
end_landing = 5000

log = [(start_takeoff, end_takeoff, "takeoff"),
       (start_landing, end_landing, "landing")]
        # and will add touch/go when footage filmed

defined_outputs = {"takeoff": takeoff_layer,
                   "landing": landing_layer}
        # will add touch/go when filmed

for start, end, label in log:                             # Teacher Spike:
    for frame in range(start, end, 10):   # Points to an event, pushes it to desired layer.
        for n in defined_outputs[label]:
            snn.add_spike((frame + 1), n, 10)

Apos = [0.0004, 0.0002, 0.0001]                     # determines how fast it learns, tightening synapses
Aneg = [-0.0002, -0.0001, -0.00005]                 # loosening synapses
snn.stdp_setup(Apos, Aneg, positive_update=True, negative_update=False)              # Sets up stdp based on values above

defined_inputs = {"takeoff": ascent_layer,
                  "landing": descent_layer}

for start, end, label in log:                             # Input Spike:
    for frame in range(start, end):   # Fake camera for when testing inputs with dummy values.
        for n in defined_inputs[label]:
            snn.add_spike((frame), n, 2.0)

descent_ids = []
for neuron in descent_layer:
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

W = snn.weight_mat()

descent_landing_before = W[np.ix_(descent_ids[:3], landing_ids)].mean()
descent_takeoff_before = W[np.ix_(descent_ids[:3], takeoff_ids)].mean()

ascent_landing_before = W[np.ix_(ascent_ids[:3], landing_ids)].mean()
ascent_takeoff_before = W[np.ix_(ascent_ids[:3], takeoff_ids)].mean()

snn.simulate(time_steps=5500)

W = snn.weight_mat()

descent_landing_after = W[np.ix_(descent_ids[:3], landing_ids)].mean()
descent_takeoff_after = W[np.ix_(descent_ids[:3], takeoff_ids)].mean()

ascent_landing_after = W[np.ix_(ascent_ids[:3], landing_ids)].mean()
ascent_takeoff_after = W[np.ix_(ascent_ids[:3], takeoff_ids)].mean()

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

