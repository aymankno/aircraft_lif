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

neuron_1a = LIF_neuron(tau=0.1, thresh=3.0, reset=0, dt=1.0/100.0)
neuron_1b = LIF_neuron(tau=0.1, thresh=1.2, reset=0, dt=1.0/100.0)
neuron_2 = LIF_neuron(tau=1, thresh=2, reset=0, dt=1.0/100.0)

takeoffs = 0
landings = 0
touch_gos = 0
y_old = None

video = "video.mp4"
video2 = "video2.mp4"
# video_3 = "videos/video3.mp4"
cap = cv2.VideoCapture("video2.mp4")

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
                if -25 <= delta_y <= -0.1:
                    if right <= 1250:
                        if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                            _, fired_1a = neuron_1a.step(abs(delta_y))
                            if fired_1a == True:
                                landings += 1
                                print("Landing")                         
                        else:
                            _, _ = neuron_2.step(1)
                if 0.1 <= delta_y <= 15:
                    if right <= 1250:
                        if (stats[best_i, cv2.CC_STAT_TOP] + stats[best_i, cv2.CC_STAT_HEIGHT]) <= 750:
                            _, fired_1b = neuron_1b.step(abs(delta_y))
                            if fired_1b == True:
                                takeoffs += 1
                                print("Takeoff")
                                neuron_1b.V = neuron_1b.reset
                            ### Does not work; no current videos with touch/go, will implement later.
                            if fired_1b == False:
                                if neuron_2.V >= 0.5:
                                    touch_gos += 1
                                    print("Touch and go")
                                    neuron_2.V = neuron_2.reset
                                    neuron_1b.V = neuron_1b.reset

    if neuron_1a.V > 0:
        neuron_1a.leak()
    if neuron_1b.V > 0:
        neuron_1b.leak()
    if neuron_2.V > 0:
        neuron_2.leak()
        
    cv2.imshow("Aircraft Counter LIF", frame)
    cv2.waitKey(33)

print(f"takeoffs: {takeoffs},  landings: {landings}, touch & go: {touch_gos}")