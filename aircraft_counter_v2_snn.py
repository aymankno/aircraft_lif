### Ayman Aghel 10/5/2026
from superneuromat import SNN
import cv2
import numpy as np

snn = SNN()

# 25 neurons each - eyes of SNN (input layers)
descent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]
ascent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(25)]

# 10 neurons each -  decision layers (output layers)
landing_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=0.1, refractory_period=10) for i in range(10)]

start_takeoff = 1500
end_takeoff = 2000

start_landing = 4500
end_landing = 5000

log = [(start_takeoff, end_takeoff, "takeoff"),
       (start_landing, end_landing, "landing")]
        # and will add touch/go when footage filmed

defined = {"takeoff": takeoff_layer,
           "landing": landing_layer}

for start, end, label in log:
    for frame in range(start, end, 10):
        for n in defined[label]:
            snn.add_spike((frame + 1), n, 10)