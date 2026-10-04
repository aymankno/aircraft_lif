from superneuromat import SNN
import cv2
import numpy as np

snn = SNN()

D = 15
descent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(D)]

A = 15
ascent_layer = [snn.create_neuron(threshold=((i+1) * 0.5)) for i in range(A)]

O = 10
# not adjusted weights
landing_layer = [snn.create_neuron(threshold=1.0, leak=1.0, refractory_period=1.0) for i in range(O)]
takeoff_layer = [snn.create_neuron(threshold=1.0, leak=1.0, refractory_period=1.0) for i in range(O)]
touch_go_layer = [snn.create_neuron(threshold=1.0, leak=1.0, refractory_period=1.0) for i in range(O)]

for pre in descent_layer:
    for post in landing_layer:
        snn.create_synapse(pre, post, weight=0.8, stdp_enabled=True)