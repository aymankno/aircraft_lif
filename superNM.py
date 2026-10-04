from superneuromat import SNN
import cv2
import numpy as np

snn = SNN()

D = 15
descent_layer = [snn.create_neuron(threshold=i) for i in range(D)]

A = 15
ascent_layer = [snn.create_neuron(threshold=i) for i in range(A)]