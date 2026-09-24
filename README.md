## aircraft_counter
Ayman Aghel - 9/14/2026

Manually coded leaky-integrate-fire network (LIF) to count aircraft and identify what stage of flight they are in: takeoff, landing, or touch and go. Designed for general aviation (GA) airports without towers to recieve grants from the Federal Aviation Administration (FAA) to improve safety. Coded exclusively with OpenCV and NumPy libraries in Python.

### Initial commit - 9/14/2026:
Currently with a working, accurate landing neuron (neuron_1a), but takeoff neuron (neuron_1b) is faulty, so is touch and go (neuron_2). In next commit, will fix these neurons and begin working on a constantly working leak mechanic. Currently, neurons only leak when calling .step(), will include a working leak mechanic at end to accurately and continously decrease voltage. Current leak mechanic does not work.

Future changes: landing has extremely larger takeoff voltages due to higher alpha values. Likely due to the fact that landing is significantly closer, therefore has larger alpha values. Will try decreasing tau and dt, increase threshold?

### 9/18/2026:
Fixed the leak mechanism. Was originally fluctuating up between 1 and 10, fixed by changing the math in leak() and including a "round" feature by defaulting self.V to 0 when below a certain voltage (0.001). Tau and dt changed, dt decreased to 0.01 (1.0 / 100.0). On test footage (video2), achieves a 100% accuracy. Added the video2 to the repository.

Future changes: Fix the go around neuron (neuron2), test on new footage. Also, may include a limit on time between events if necessary--like a minimum of 60 seconds between each takeoff, landing, and touch/go.

### 9/22/2026:
Added a go-around mechanism, added another neuron (neuron_4), and re-named all of the neurons to be neuron 1-4. The go-around architecture I imagine is first an ascent neuron (neuron_4) being fired, then running electricity to the descent neuron (neuron_3), and only passing if neuron_3 has any charge from a recent descent. If there is no charge, it won't pass. I chose this architecture because my initial approach was flawed and would not have held up under pressure. Initially, I was going to use neuron_2 (the previous, sole touch/go neuron) and have it build charge exclusively on the descent (when -25 < alpha < -1) and count a go-around when any positive alpha was detected. This approach would not have held up due to noise, so I decided to use the alternative architecture. Minor changes: changed takeoff neuron (neuron_1) thresh. Neuron 1 and 2 are expieriencing 100% accuracy on video1, will get more videos.

Future changes: Test the touch/go neurons (neuron_3 + neuron_4) on touch/go footage and change paramaters as nessescary. Still considering a time limit between events; likely with a dedicated neuron that charges when any activity is occuring (or counted). Very likely.

### 9/23/2026
Major improvements: Added neuron_5: a time neuron from the LIF_neuron class dedicated to ensuring no events happen abnormally close together. When an event occurs, the neuron is reset to 1, where it then leaks for around 80 seconds until reaching below 0.1. Once below 0.1, events can be triggered and counted. This allows for a decreased threshold in the future; repeated triggers will be much less common.

Future changes: Set touch/go neuron's paramaters. Looked at footage today, but found no usable footage of go-arounds, especially at GA airports. Will still continue looking for footage of go-arounds, takeoffs, and landings to test syntax + concept. Also: polish code and improve syntax, readability (especially in the repeated if statements before neurons are fired) of aircraft_counter.py. Fix grammar, organize thoughts in this README.