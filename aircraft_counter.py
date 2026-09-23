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

    def step(self, I_input):
        did_fire = False
        V_new = self.V + (self.dt / self.tau) * ((-1 * self.V) + I_input)
        if V_new > self.thresh:
            did_fire = True
            V_new = self.reset
        self.V = V_new

        return V_new, did_fire

    def leak(self):
        if self.V < 0.001:
            self.V = 0
        V_new = self.V + (self.dt / self.tau) * ((-1 * self.V))
        self.V = V_new

def get_slope(y_old, y_new):
    delta_y = y_old - y_new
    return delta_y

# neuron_1: Takeoff
neuron_1 = LIF_neuron(tau=0.1, thresh=1.5, reset=0, dt=1.0/100.0)

# neuron_2: Landing
neuron_2 = LIF_neuron(tau=0.1, thresh=3.0, reset=0, dt=1.0/100.0)

# neuron_3: touch/go LAYER 1 - descent
neuron_3 = LIF_neuron(tau=1, thresh=0.1, reset=0, dt=1.0/100.0)

# neuron_4: touch/go LAYER 2 - ascent
neuron_4 = LIF_neuron(tau=1, thresh=0.1, reset=0, dt=1.0/100.0)

takeoffs = 0
landings = 0
touch_gos = 0
y_old = None

video1 = "video.mp4"
video2 = "videos/video2.mp4"
#video_3 = "videos/video3.mp4"
cap = cv2.VideoCapture(video2)

prev_frame = None
y_old = None
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
                if right <= 1250:
                    if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                        if -25 <= delta_y <= -0.1:
                            _, fired_2 = neuron_2.step(abs(delta_y))
                            _, fired_3 = neuron_3.step(abs(delta_y))
                            if fired_2 == True:
                                landings += 1
                                print("Landing") 
                                neuron_2.V = neuron_2.reset                        

                if 0.1 <= delta_y <= 15:
                    if right <= 1250:
                        if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                            _, fired_1 = neuron_1.step(abs(delta_y))
                            _, fired_4 = neuron_4.step(abs(delta_y))
                            if fired_1 == True:
                                takeoffs += 1
                                print("Takeoff")
                                neuron_1.V = neuron_1.reset
                            ### go-around is experimental mechanism
                            if fired_4 == True:
                                if neuron_3.V > 0.1:
                                    touch_gos += 1
                                    print("Touch and go")
                                    neuron_3.V = neuron_3.reset
                                    neuron_4.V = neuron_4.reset

    if neuron_1.V > 0:
        neuron_1.leak()
    if neuron_2.V > 0:
        neuron_2.leak()
    if neuron_3.V > 0:
        neuron_3.leak()
    if neuron_4.V > 0:
        neuron_4.leak()
        
    cv2.imshow("Aircraft Counter LIF", frame)
    cv2.waitKey(33)

print(f"takeoffs: {takeoffs},  landings: {landings}, touch & go: {touch_gos}")