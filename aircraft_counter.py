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
        self.alpha = 0.6

        self.old_input = 0


    def step(self, I_input):
        did_fire = False
        smooth_input = ((self.alpha * I_input) + ((1 - self.alpha) * self.old_input))
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
def within_paramaters(min, max):
    is_within = False
    if min <= delta_y <= max:
        if right <= 1250:
            if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                is_within = True
    return is_within

# neuron_1: Takeoff
neuron_1 = LIF_neuron(tau=0.1, thresh=1.5, reset=0, dt=1.0/100.0)

# neuron_2: Landing
neuron_2 = LIF_neuron(tau=2.5, thresh=0.9, reset=0, dt=1.0/100.0)

# neuron_3: touch/go LAYER 1 - descent. PARAMATERS WILL NEED ADJUSTMENT
neuron_3 = LIF_neuron(tau=5.0, thresh=0.1, reset=0, dt=1.0/100.0)

# neuron_4: touch/go LAYER 2 - ascent. PARAMATERS WILL NEED ADJUSTMENT
neuron_4 = LIF_neuron(tau=1.0, thresh=20.0, reset=0, dt=1.0/100.0)

# neuron_5: time between events. reset = voltage after event
neuron_5 = LIF_neuron(tau=5.0, thresh=0.05, reset=1.0, dt=1.0/100.0)


takeoffs = 0
landings = 0
touch_gos = 0
y_old = None

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
            else:
                top_left = (stats[best_i, cv2.CC_STAT_LEFT], stats[best_i, cv2.CC_STAT_TOP])
                right = stats[best_i, cv2.CC_STAT_LEFT] + stats[best_i, cv2.CC_STAT_WIDTH]
                bottom_right = (right, stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT])
                delta_y = get_slope(y_old, centroid)
                y_old = centroid
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 20, cv2.LINE_8)

                descent_paramaters = within_paramaters(-25, -0.1)
                ascent_paramaters = within_paramaters(0.1, 15)

                if descent_paramaters == True:
                    _, fired_2 = neuron_2.step(abs(delta_y))
                    _, fired_3 = neuron_3.step(abs(delta_y))
                    neuron_1.leak()
                    neuron_4.leak()
                    neuron_5.leak()
                    if fired_2 == True:
                        if neuron_5.V < neuron_5.thresh:
                            landings += 1
                            print("Landing") 
                            neuron_5.V = neuron_5.reset
                            neuron_2.V = neuron_2.reset                        

                elif ascent_paramaters == True:
                    _, fired_1 = neuron_1.step(abs(delta_y))
                    _, fired_4 = neuron_4.step(abs(delta_y))
                    neuron_2.leak()
                    neuron_3.leak()
                    neuron_5.leak()
                    if fired_1 == True:
                        if neuron_5.V < neuron_5.thresh:
                                    takeoffs += 1
                                    print("Takeoff")
                                    neuron_5.V = neuron_5.reset
                                    neuron_1.V = neuron_1.reset

                    ### go-around is experimental mechanism; concept/syntax looks good, paramaters untested
                    if fired_4 == True:
                            if neuron_3.V > 0.1:
                                if neuron_5.V < neuron_5.thresh:
                                    touch_gos += 1
                                    print("Touch and go")
                                    neuron_5.V = neuron_5.reset
                                    neuron_3.V = neuron_3.reset
                                    neuron_4.V = neuron_4.reset
                else:
                    neuron_1.leak()
                    neuron_2.leak()
                    neuron_3.leak()
                    neuron_4.leak()
                    neuron_5.leak()

    print(neuron_2.V)
    cv2.imshow("Aircraft Counter LIF", frame)
    cv2.waitKey(33)

print(f"takeoffs: {takeoffs},  landings: {landings}, touch & go: {touch_gos}")