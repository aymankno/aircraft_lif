# aircraft_counter
Ayman Aghel - 9/14/2026

V1 = aircraft_counter.py - Custom LIF network coded manually in Python

V2 = ac_superneuromat.py - SuperNeuroMAT spiking neural network

Neuromorphic-inspired network to count and classify takeoffs, landings, and touch-and-goes at untowered airports from video. Aims to achieve high accuracy when counting & classifying aircraft movements at airports for grant applications to improve safety and maintainence. Initially a manually coded custom system of five leaky integrate-and-fire (LIF) neurons coded with NumPy and OpenCV with a heavy dependency on if statements, currently an SNN coded with the SuperNeuroMAT simulator in Python.

Latest changes (10/8/2026):
Filmed footage of 40 movements at DKX, raised questions including counting of helicopters, training certain things as "nothing" including geese, maybe go-arounds? Considering using acoustic detection of aircraft to turn on camera for superior energy efficiency despite DKX proximity to boats, trains nearby.

## License
[PolyForm Noncommercial 1.0.0](LICENSE): free for personal, educational, and research use. Commercial use requires permission. Contact: aymanaghel@gmail.com

## Changelog

### Initial commit - 9/14/2026:
Currently with a working, accurate landing neuron (neuron_1a), but takeoff neuron (neuron_1b) is faulty, so is touch and go (neuron_2). In next commit, will fix these neurons and begin working on a constantly working leak mechanic. Currently, neurons only leak when calling .step(), will include a working leak mechanic at end to accurately and continuously decrease voltage. Current leak mechanic does not work.

Future changes: landing has extremely larger takeoff voltages due to higher delta values. Likely due to the fact that landing is significantly closer, therefore has larger delta values. Will try decreasing tau and dt, increase threshold?

### Fixed leak mechanism - 9/18/2026:
Fixed the leak mechanism. Was originally fluctuating up between 1 and 10, fixed by changing the math in leak() and including a "round" feature by defaulting self.V to 0 when below a certain voltage (0.001). Tau and dt changed, dt decreased to 0.01 (1.0 / 100.0). Added the video2 to the repository.

Future changes: Fix the go around neuron (neuron2), test on new footage. Also, may include a limit on time between events if necessary--like a minimum of 60 seconds between each takeoff, landing, and touch/go.

### Added touch & go mechanism, new neuron - 9/22/2026:
Added a go-around mechanism, added another neuron (neuron_4), and re-named all of the neurons to be neuron 1-4. The go-around architecture I imagine is first an ascent neuron (neuron_4) being fired, then running electricity to the descent neuron (neuron_3), and only passing if neuron_3 has any charge from a recent descent. If there is no charge, it won't pass. I chose this architecture because my initial approach was flawed and would not have held up under pressure. Initially, I was going to use neuron_2 (the previous, sole touch/go neuron) and have it build charge exclusively on the descent (when -25 < delta < -1) and count a go-around when any positive delta was detected. This approach would not have held up due to noise, so I decided to use the alternative architecture. Minor changes: changed takeoff neuron (neuron_1) thresh. Neuron 1 and 2 are experiencing 100% accuracy on video1, will get more videos.

Future changes: Test the touch/go neurons (neuron_3 + neuron_4) on touch/go footage and change parameters as necessary. Still considering a time limit between events; likely with a dedicated neuron that charges when any activity is occurring (or counted). Very likely.

### Added new time neuron to prevent bursts of events - 9/23/2026
Major improvements: Added neuron_5: a time neuron from the LIF_neuron class dedicated to ensuring no events happen abnormally close together. When an event occurs, the neuron is reset to 1, where it then leaks for around 80 seconds until reaching below 0.1. Once below 0.1, events can be triggered and counted. This allows for a decreased threshold in the future; repeated triggers will be much less common.

Future changes: Set touch/go neuron's parameters. Looked at footage today, but found no usable footage of go-arounds, especially at GA airports. Will still continue looking for footage of go-arounds, takeoffs, and landings to test syntax + concept. Also: polish code and improve syntax, readability (especially in the repeated if statements before neurons are fired) of aircraft_counter.py. Fix grammar, organize thoughts in this README.

### Improved readability of code, explored other videos - 9/30/2026
Created a within_parameters function to improve readability. Function passes minimum and maximum delta_y values to filter blobs by location and delta to determine which blobs should trigger a step() for their respective neuron. Function currently fit to video2, but allows for easy changing between videos. Investigated other videos to determine accuracy of custom LIF network, resulted in a fairly high accuracy yet an issue with camera focus. Video would blur in certain videos at random times, trigger neuron_5, and thus prevent actual movements from triggering; like counting a landing during a takeoff, etc. In addition, landings would be too distant from the camera in certain videos for OpenCV to detect a plane as a moving object (a blob). This can likely be fixed by camera placement during field testing rather than code.

Future changes: implement a fix for blobs on screen: debris on camera, focus, etc. Go to local airfields to self-film footage to determine optimal location for deployment and to evaluate what changes are needed to the system.

### Added smoothing to inputs, further organized code - 10/2/2026
Added a smoothing feature to inputs to combat issue with blur and noise in footage. Seems to be effective in initial testing, but definitely could use fixing later and with new footage. Still consistently receiving a good score on the test footage. Minor changes: improved readability of code, changed parameters of landing neuron.

Future changes: film footage at a GA field to fit parameters to. Find new ways to combat blurring issue, possibly with more neurons.

### Began work with SuperNeuroMAT - 10/4/2026
SuperNeuroMAT is a Python package widely used by the neuromorphic community to get weights for neuromorphic models, like SNNs. Although aircraft_counter.py is comprised of just five neurons, SuperNeuroMAT makes it possible to replace the singular neurons used with layers of neurons for superior accuracy. The primary purpose of this project, initially, was to further my understanding in neuromorphic computation, LIFs, and SNNs, but now this project is allowing me to learn how to use common neuromorphic libraries in the neuromorphic community like SuperNeuroMAT. SuperNeuroMAT will allow for the testing of parameters and training of neurons. An initial commit can be found at superNM.py.

Changes to aircraft_counter.py: Removed several lines of unnecessary code. Fixed bug in go_around block of code; incorrect indentation.

Future changes: Film footage at GA fields to train SuperNeuroMAT layers of neurons on, further understanding of SuperNeuroMAT.

### Fundamental changes to aircraft_counter.py & repository - 10/5/2026
Renamed superNM.py to ac_superneuromat.py for readability. "ac", short for aircraft_counter. Added teacher spike, synapses, and initialized STDP to prepare for test run with fake values. Added comments throughout for context when coming back to project (new library + concept).

Important changes to aircraft_counter.py (fundamental): While working through the conversion from aircraft_counter.py to ac_superneuromat.py, it became evident that in the event of a go-around, under current architecture, a go-around would never been counted and instead be counted as a landing due to the fact that neurons act independently of each other. Resulted in all neurons being repurposed in some way, like neuron_1 and 2 being changed to ascent/descent, and neuron_3-4 becoming time neurons. Furthermore, while doing independent reading on SNNs, I came to the realization that aircraft_counter.py is not a manually coded SNN and is instead a custom system of LIFs with a heavy dependency on if statements with Python. This means that aircraft_counter.py will be used as reference in the future and likely will not be used for testing of parameters both due to the lack of synapses and of neurons (singular neurons rather than systems of neurons). Furthermore, this repository, aircraft_counter, will now have a heavy focus on ac_superneuromat.py for this reason.

### Trained & tested ac_superneuromat.py on fake values successfully - 10/6/2026
Added input spike to training and completed successfully, wrote full test and completed successfully. During training, synapse weights would change on average by 0.0319 for descent-landing synapses and ascent-takeoff neurons but remain the same for descent-takeoff and ascent-landing neurons; an expected result. For testing, manually fed correct input neuron layers spikes within given ranges of takeoffs and landings, spiked 212 times for landing-landing and 243 for takeoff-takeoff, while neuron layers without correlation (takeoff-landing, landing-takeoff) would have no spikes. These are the first real results to come from ac_superneuromat.py; aircraft_counter coded with SuperNeuroMAT.

Notes: Untaught output neurons were firing on their own during training, so initial weights were lowered to 0.01-0.02 to keep input per frame below leak. Test input was identical to training input (a constant 2.0 input w/o noise), confirming that the pipeline works, not that it has real-world accuracy. Aneg was draining every synapse for every step, so it was disabled in STDP setup. For add_spike, time was relative to the current step, so testing frames had to be set to frame - 5500; 5500 time steps during training already occurred.

Future changes: Will begin testing on video and train as needed. Plans to film at local GA field set, expecting to film as many aircraft as possible with a  goal of filming 30 movements and at least 10 touch & goes. Will log start_time, end time, and type of movement for each movement.

### Tested SuperNeuroMAT model on onlne footage, minor changes to LIF - 10/7/2026
Copied the OpemCV initialization from aircraft_counter to ac_superneuromat.py with changes as needed. Connected delta_y to input neurons and copied over within_parameters (footage is online with graphics on right-hand side). After running, discovered large differences between takeoff and landing spikes: 19 takeoff spikes and 117 landing spikes. Then, changed synapses' weights to a new range: 0.0035 - 0.0084 to ensure that neurons firing do not overpower leak (more neurons firing for descent layer with new training). With new weights, although accomodating the descent_layer,  leak was over-powered in ascent_layer neurons. To fix, I trained with more synthetic data. Worked, then tested to make sure neuron layers were spiking on correct layers (they were), and wrapped up.

Notes: Output neurons have a large imbalance: 19 takeoff spikes v. 117 landing spikes. Calculated avg. delta for ascent & descent (abs. value), set as static input during synthetic training and testing. After changing, though, a similar issue from yesterday, intial weights lowered to prevent leaking from overpowering input. Trained with more synthetic data to fix, then made sure neurons spiking on correct events (correct times). All systems green!

Future changes: Will be filming footage tomorrow at airfield for new test footage at a target site for implementation of aircraft_counter, allowing for more fitting, real training AND testing data, and footage of touch & goes.

### Filmed footage at DKX, created frame_check.py - 10/8/2026
Filmed footage at Knoxville Downtown Island airport from Island Home Park in Knoxville, Tennessee. 40 movements recorded, exactly 10 touch & goes recorded, and 2 go-arounds recorded. Landings and takeoffs are near-even. While filming, I realized that due to the distance from the camera to the takeoofs, OpenCV will likely not recognize takeoffs as blobs to detect centroids. Therefore, a new feature of the SNN will be implemented: a split set of input neurons for a zoomed in area above the runway where takeoffs fly, and one for a certain portion of the screen where takeoffs land. This also means seperate OpenCV uses. Although not implemented, this is worth noting. Furthermore, created frame_check.py to accurately label and match events to frame number. However, there is no CSV due to bugs during code and limited time available to me.

Notes: Helicopters flying above, not sure how I will count those in the future, worth noting. Go-around will be tricky to train against touch/goes, two go-arounds can be used to train as "nothing" category, also geese? Could crop touch/goes to get more training and testing data in half to just show as landing or takeoff, although likely not needed due to abundance of events. Required counting actually fluctuates airport to airport and for different grant ptojects, planning to send email to DKX airport manager reguarding future projects at airport.

Future changes: Use frame_check or different method to get frame #s for training spikes. Upload some of the filmed videos to YouTube or other file saving platform. Get usable audio and consider using to train model on acoustic signs for helicopters? Further investigate what needs to be counted at airports.

### Full training loop, matched inputs, additional input layers, seperate on-screen zones of interest, "reverse" mechanism - 10/9/2026
Wrote the full training loop for video9. While changing blocks from online/synthetic to the video filmed, it became apparent that more input neurons will be needed now and in the future. The first would be size neurons that are spiked by the size of a blob on screen. In the future, these neurons could be replaced by growing/shrinking neurons to determine direction of traffic; more on this later. Additionaly, OpenCV would not detect the small, distant aircraft taking off. This led to the implementation of a "dual" screen system, defined as "track" which takes the mask, box, minimum area and maximum area to zoom in on certain areas to count aircraft. In the future, layers of neurons could bridge the two layers to determine if an event was a touch & go, as the two seperate regions allow for distinct ascent/descent detection. For instance, the two seperate areas each are currently exclusively allowed to charge designated input neurons, since all footage filmed (as of now) was in a singular direction of traffic: away from the camera. This means that when there is a change in direction of traffic, the counter would be at a disadvantage to count aircraft, having to rely on the small intersection of the layers. This supports the idea of a shrinking/growing input layer which can fire a "reverse" layer of neurons which would reverse the direction of traffic and change the on-screen zones to only be able to charge the opposite input layer (although, only that layer). 

Notes: Fundamental change from synthetic to video training required a lot of replacing, significantly more than when synthetic data was changed to the median delta_y of an online video. More input layers could be very beneficial to the accuracy of the model. Initially considered growing/shrinking neuron layer, but realized that it would be useless under current architecture; all events shrink (one direction). Would be useful for a reverse-traffic feature, although not yet possible due to lack of footage. Perhaps reversing videos? Additionally, a time_off_screen neuron layer that recieves input during the time neither zone has spikes would be beneficial to a touch_go mechanism, especially considering it would not accidentally count go-arounds. Ran tests to get optimal minimum and maximum areas, deltas for blobs to be counted and spike input layers, added synapses to match the two additonal input layers (just size, not shrinking/growing currently). 

Future changes: Write a test on a different video filmed, determine if a size neuron is worth keeping or if a filter is enough, add growing/shrinking for "reverse" traffic (trained/tested on videos reversed?), consider implentation of aircraft classification in the future, needed for large grants. Consider outputs as popular aircraft make/model and then automatically classify, train on the models themselves, perhaps tail # capturing?

## Acknowledgements:

```bibtex
@inproceedings{date2023superneuro,
  title     = {SuperNeuro: A fast and scalable simulator for neuromorphic computing},
  author    = {Date, Prasanna and Gunaratne, Chathika and Kulkarni, Shruti R. and Patton, Robert and Coletti, Mark and Potok, Thomas},
  booktitle = {Proceedings of the 2023 International Conference on Neuromorphic Systems},
  pages     = {1--4},
  year      = {2023}
}

@misc{date2026superneuromat,
  title         = {SuperNeuroMAT: An Efficient Matrix-based Simulator for Spiking Neural Networks},
  author        = {Date, Prasanna and Zhu, Kevin and Kulkarni, Shruti and Gautam, Ashish and
                   Gunaratne, Chathika and Patton, Robert and Nitzsche, Tyler and Mulet, Ian and
                   Johnson-Scott, Zachary and Helms, Addison and Rowden, Duncan and
                   Weston, Simon and Parsa, Maryam and Schuman, Catherine and Potok, Thomas},
  year          = {2026},
  eprint        = {2608.08479},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  doi           = {10.48550/arXiv.2608.08479}
}