# aircraft_counter
Ayman Aghel - 9/14/2026

V1 = aircraft_counter.py - Custom LIF network coded manually in Python      
V2 = ac_superneuromat.py - SuperNeuroMAT spiking neural network        

Check out the changelog (40+ entries), now at [CHANGELOG.md](./CHANGELOG.md) !       
video1-10 - [YouTube Link](https://www.youtube.com/playlist?list=PLKrqNoJmfkQE)      

Neuromorphic-inspired network to count and classify takeoffs, landings, and touch-and-goes at untowered airports from video. Aims to achieve high accuracy when counting & classifying aircraft movements at airports for grant applications to improve safety and maintainence. Initially a manually coded custom system of five leaky integrate-and-fire (LIF) neurons coded with NumPy and OpenCV with a heavy dependency on if statements, currently an SNN coded with the SuperNeuroMAT simulator in Python.
Note: Elite energy effiency is not yet tested; currently simulated completely with SuperNeuroMAT.

Latest changes (11/18) - Dynamic zones active, accurate. 
Additional changes to tiles on network, used in testing and, for now, working great! Currently needs a seeded initial zones, which then gets moved as needed with a neighbor system. Not the best in the long term, so I am planning to replace it with a model that can start counting from scratch; plug & play. Also, created blocks on tiles that are noisy and likely shrubs, model currently flags water consistently (good thing!).

## License
[PolyForm Noncommercial 1.0.0](LICENSE): free for personal, educational, and research use. Commercial use requires permission. Contact: aymanaghel@gmail.com

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