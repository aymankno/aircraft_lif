### Ayman Aghel 9/6/26
import numpy as np
import cv2

class LIF_neuron:
    def __init__(self, tau, thresh, reset, dt):
        self.V = 0
        self.tau = tau
        self.thresh = thresh
        self.reset = reset
        self.dt = dt
        self.delta = 0.8
        self.old_input = 0

    def step(self, I_input):
        did_fire = False
        smooth_input = ((self.delta * I_input) + ((1 - self.delta) * self.old_input))
        V_new = self.V + (self.dt / self.tau) * ((-1 * self.V) + smooth_input)
        if V_new > self.thresh:
            did_fire = True
            V_new = self.reset
        self.V = V_new
        old_input = smooth_input

        return V_new, did_fire

    def leak(self):
        if self.V < 0.05:
            self.V = 0
        V_new = self.V + (self.dt / self.tau) * ((-1 * self.V))
        self.V = V_new

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

# designed for certain video used for testing with graphic on right side
def within_parameters(min, max):
    is_within = False
    if min <= delta_y <= max:
        if right <= 1250:
            if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                if neuron_5.V < neuron_5.thresh:
                    is_within = True
    return is_within

# neuron_1: Ascent neuron - INPUT NEURON - positive delta_y of centroid
neuron_1 = LIF_neuron(tau=0.1, thresh=1.5, reset=0, dt=1.0/100.0)

# neuron_2: Descent neuron - INPUT NEURON - negative delta_y of centroid
neuron_2 = LIF_neuron(tau=2.5, thresh=1.0, reset=0, dt=1.0/100.0)

# neuron_3: Landing neuron (NEW USE) - TIME NEURON - from descent trigger --> landing trigger (goal)
neuron_3 = LIF_neuron(tau=5.0, thresh=0.05, reset=1.0, dt=1.0/100.0) # PARAMETERS WRONG; FOR REFERENCE

# neuron_4: Touch/go neuron for ascent - TIME NEURON - from descent trigger --> touch/go trigger (goal)
neuron_4 = LIF_neuron(tau=3.0, thresh=0.05, reset=1.0, dt=1.0/100.0) # PARAMETERS WRONG; FOR REFERENCE

# neuron_5: time between events - TIME NEURON - from event trigger --> next event allowed to trigger
neuron_5 = LIF_neuron(tau=5.0, thresh=0.05, reset=1.0, dt=1.0/100.0)


takeoffs = 0
landings = 0
touch_gos = 0
y_old = None
fired_2 = False

video1 = "video.mp4"
video2 = "/Users/aymanaghel/Desktop/LIF/aircraft_lif/videos/video2.mp4"
#video_3 = "videos/video3.mp4"
cap = cv2.VideoCapture(video2)

prev_frame = None
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray
    mask = backSub.apply(gray)
    prev_frame = gray
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        best_i = np.argmax(stats[1: , cv2.CC_STAT_AREA]) + 1
        if 1000 <= stats[best_i, cv2.CC_STAT_AREA] <= 25000:
            centroid = (stats[best_i, cv2.CC_STAT_TOP] + (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT])) / 2
            if y_old is None:
                y_old = centroid

            top_left = (stats[best_i, cv2.CC_STAT_LEFT], stats[best_i, cv2.CC_STAT_TOP])
            right = stats[best_i, cv2.CC_STAT_LEFT] + stats[best_i, cv2.CC_STAT_WIDTH]
            bottom_right = (right, stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT])
            delta_y = get_slope(y_old, centroid)
            y_old = centroid
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 20, cv2.LINE_8)

            descent_parameters = within_parameters(-25, -0.1)
            ascent_parameters = within_parameters(0.1, 15)

            if descent_parameters == True:
                _, fired_2 = neuron_2.step(abs(delta_y))
                neuron_1.leak()
                neuron_4.leak()
                neuron_5.leak()
                if fired_2 == True:
                    print("Descent detected - nothing counted") 
                    neuron_5.V = neuron_5.reset
                    neuron_3.V = neuron_3.reset
                    neuron_2.V = neuron_2.reset                        

            elif ascent_parameters == True:
                _, fired_1 = neuron_1.step(abs(delta_y))
                _, fired_4 = neuron_4.step(abs(delta_y))
                neuron_2.leak()
                neuron_5.leak()
                if fired_1 == True:
                    takeoffs += 1
                    print("Takeoff")
                    neuron_5.V = neuron_5.reset
                    neuron_1.V = neuron_1.reset

            else:
                neuron_1.leak() # input
                neuron_2.leak() # input
                neuron_3.leak() # time
                neuron_4.leak() # time
                neuron_5.leak() # time

    # descent block - landings & touch/go
    if fired_2 == True:
        if neuron_5.V < neuron_5.thresh:
            if neuron_3.V < neuron_3.thresh:
                print("Landing confirmed")
                landings += 1
                neuron_5.V = neuron_5.reset
                fired_2 = False
            elif neuron_4.V < neuron_4.thresh:
                print("Touch/go confirmed")
                touch_gos += 1
                neuron_5.V = neuron_5.reset
                fired_2 = False

    cv2.imshow("Aircraft Counter LIF", frame)
    cv2.waitKey(33)

print(f"takeoffs: {takeoffs},  landings: {landings}, touch & go: {touch_gos}")