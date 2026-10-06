## aircraft_counter
Ayman Aghel - 9/14/2026

Custom LIF network of five neurons to count aircraft and classify and sort by stage of flight: takeoff, landing, and touch & go. Initially coded in Python manually with OpenCV and NumPy, now with SuperNeuroMAT implementation.

### Initial commit - 9/14/2026:
Currently with a working, accurate landing neuron (neuron_1a), but takeoff neuron (neuron_1b) is faulty, so is touch and go (neuron_2). In next commit, will fix these neurons and begin working on a constantly working leak mechanic. Currently, neurons only leak when calling .step(), will include a working leak mechanic at end to accurately and continously decrease voltage. Current leak mechanic does not work.

Future changes: landing has extremely larger takeoff voltages due to higher delta values. Likely due to the fact that landing is significantly closer, therefore has larger delta values. Will try decreasing tau and dt, increase threshold?

### Fixed leak mechanism - 9/18/2026:
Fixed the leak mechanism. Was originally fluctuating up between 1 and 10, fixed by changing the math in leak() and including a "round" feature by defaulting self.V to 0 when below a certain voltage (0.001). Tau and dt changed, dt decreased to 0.01 (1.0 / 100.0). On test footage (video2), achieves a 100% accuracy. Added the video2 to the repository.

Future changes: Fix the go around neuron (neuron2), test on new footage. Also, may include a limit on time between events if necessary--like a minimum of 60 seconds between each takeoff, landing, and touch/go.

### Added go-around mechanism and a new neuron - 9/22/2026:
Added a go-around mechanism, added another neuron (neuron_4), and re-named all of the neurons to be neuron 1-4. The go-around architecture I imagine is first an ascent neuron (neuron_4) being fired, then running electricity to the descent neuron (neuron_3), and only passing if neuron_3 has any charge from a recent descent. If there is no charge, it won't pass. I chose this architecture because my initial approach was flawed and would not have held up under pressure. Initially, I was going to use neuron_2 (the previous, sole touch/go neuron) and have it build charge exclusively on the descent (when -25 < delta < -1) and count a go-around when any positive delta was detected. This approach would not have held up due to noise, so I decided to use the alternative architecture. Minor changes: changed takeoff neuron (neuron_1) thresh. Neuron 1 and 2 are expieriencing 100% accuracy on video1, will get more videos.

Future changes: Test the touch/go neurons (neuron_3 + neuron_4) on touch/go footage and change paramaters as nessescary. Still considering a time limit between events; likely with a dedicated neuron that charges when any activity is occuring (or counted). Very likely.

### Added new time neuron to prevent bursts of events - 9/23/2026
Major improvements: Added neuron_5: a time neuron from the LIF_neuron class dedicated to ensuring no events happen abnormally close together. When an event occurs, the neuron is reset to 1, where it then leaks for around 80 seconds until reaching below 0.1. Once below 0.1, events can be triggered and counted. This allows for a decreased threshold in the future; repeated triggers will be much less common.

Future changes: Set touch/go neuron's paramaters. Looked at footage today, but found no usable footage of go-arounds, especially at GA airports. Will still continue looking for footage of go-arounds, takeoffs, and landings to test syntax + concept. Also: polish code and improve syntax, readability (especially in the repeated if statements before neurons are fired) of aircraft_counter.py. Fix grammar, organize thoughts in this README.

### Improved readability of code, explored other videos - 9/30/2026
Created a within_paramaters function to improve readability. Function passes minimum and maximum delta_y values to filter blobs by location and delta to determine which blobs should trigger a step() for their respective neuron. Function currently fit to video2, but allows for easy changing between videos. Investigated other videos to determine accuracy of custom LIF network, resulted in a fairly high accuracy yet an issue with camera focus. Video would blur in certain videos at random times, trigger neuron_5, and thus prevent actual movements from triggering; like counting a landing during a takeoff, etc. In addition, landings would be too distant from the camera in certain videos for OpenCV to detect a plane as a moving object (a blob). This can likely be fixed by camera placement during field testing rather than code.

Future changes: implement a fix for blobs on screen: debris on camera, focus, etc. Go to local airfields to self-film footage to determine optimal location for deployment and to evaluate what changes are needed to the system.

### Added smoothing to inputs, further organied code - 10/2/2026
Added a smoothing feature to inputs to combat issue with blur and noise in footage. Seems to be effective in initial testing, but definitely could use fixing later and with new footage. Still consistently recieving a good score on the test footage. Minor changes: improved readability of code, changed paramaters of landing neuron.

Future changes: film footage at a GA field to fit paramaters to. Find new ways to combat blurring issue, possibly with more neurons.

### Began work with SuperNeuroMAT - 10/4/2026
SuperNeuroMAT is a Python package widely used by the neuromorphic community to get weights for neuromorphic models, like SNNs. Although aircraft_counter.py is comprised of just five neurons, SuperNeuroMAT makes it possible to replace the singular neurons used with layers of neurons for superior accuracy. The primary purpose of this project, initially, was to further my understanding in neuromorphic computation, LIFs, and SNNs, but now this project is allowing me to learn how to use common neuromorphic libraries in the neuromorphic community like SuperNeuroMAT. SuperNeuroMAT will allow for the testing of paramaters and training of neurons. An initial commit can be found at superNM.py.

Changes to aircraft_counter.py: Removed several lines of unnecessary code. Fixed bug in go_around block of code; incorrect indentation.

Future changes: Film footage at GA fields to train SuperNeuroMAT layers of neurons on, further understanding of SuperNeuroMAT.

### Fundamental changes to aircraft_counter.py & repository - 10/5/2026
Renamed superNM.py to ac_superneuromat.py for readability. "ac", short for aircraft_counter. Added teacher spike, synapses, and initialized sdtp to prepare for test run with fake values. Added comments throughout for context when coming back to project (new library + concept).

Important changes to aircraft_counter.py (fundamental): While working through the conversion from aircraft_counter.py to ac_superneuromat.py, it became evident that in the event of a go-around, under current architecture, a go-around would never been counted and instead be counted as a landing due to the fact that neurons act independently of eachother. Resulted in all neurons being repurposed in some way, like neuron_1 and 2 being changed to ascent/descent, and neuron_3-4 becoming time neurons. Furthermore, while doing independent reading on SNNs, I came to the realization that aircraft_counter.py is not a manually coded SNN and is instead a custom system of LIFs with a heavy dependency on if statements with Python. This means that aircraft_counter.py will be used as reference in the future and likely will not be used for testing of paramaters both due to the lack of synapses and of neurons (singular neurons rather than systems of neurons). Furthermore, this repository, aircraft_counter, will now have a heavy focus on ac_superneuromat.py for this reason.

## Acknowledgements:

```bibtex
@inproceedings{date2023superneuro,
  title     = {SuperNeuro: A fast and scalable simulator for neuromorphic computing},
  author    = {Date, Prasanna and Gunaratne, Chathika and Kulkarni, Shruti R. and Patton, Robert and Coletti, Mark and Potok, Thomas},
  booktitle = {Proceedings of the 2023 International Conference on Neuromorphic Systems},
  pages     = {1--4},
  year      = {2023}
}