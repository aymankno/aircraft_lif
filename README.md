Ayman Aghel - 9/14/2026

# aircraft_counter

A Leaky Integrate Fire network  (LIF) to count aircraft and identify what stage of flight they are in: Takeoff, Landing, or Touch and Go. Designed for future use at General Aviation (GA) airports without towers for grants from the Federal Aviation Administration.

## Initial commit - 9/14/2026

Currently with a working, accurate landing neuron (neuron_1a), but takeoff neuron (neuron_1b) is faulty, so is touch and go (neuron_2). In next commit, will fix these neurons and begin working on a constantly working leak mechanic. Currently, neurons only leak when calling .step(), will include a working leak mechanic at end to accurately and continously decrease voltage. Current leak mechanic does not work.

Future changes: landing has extremely larger takeoff voltages due to higher alpha values. Likely due to the fact that landing is significantly closer, therefore has larger alpha values. Will try decreasing tau and dt, increase threshold?
