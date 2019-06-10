import cv2 as cv
import pygame.midi
import time

def init_pygame():
    pygame.midi.init()
    player = pygame.midi.Output(0)
    player.set_instrument(49 - 1)
    return player


def get_line_points(width, height, linesCount):
    top_starting_point = width // 5
    top_distance = 3 * width // 5 // (linesCount - 1)
    bottom_starting_point = width // 3
    bottom_distance = bottom_starting_point // (linesCount - 1)

    return [
        (
            (top_starting_point + i * top_distance, 0),
            (bottom_starting_point + i * bottom_distance, height)
        ) for i in range(linesCount)
    ]



class HarpString:
    def __init__(self, p0, p1, ndx, midi):
        self.x0, self.y0 = p0
        self.x1, self.y1 = p1
        self.p0 = p0
        self.p1 = p1
        self.A = 1
        self.B = (self.x1 - self.x0) / (self.y0 - self.y1)
        self.C = -(self.x0 + (self.x1 - self.x0) / (self.y0 - self.y1) * self.y0)
        self.note = 60 + ndx * 2
        self.midi = midi
        self.is_playing = False

    def distance(self, point):
        x0, y0 = point
        return abs(self.A * x0 + self.B * y0 + self.C) / pow(self.A**2 + self.B**2, 0.5)


    def play_sound(self):
        if not self.is_playing:
            self.is_playing = True
            self.midi.note_on(self.note, 127)

    def stop_sound(self):
        if self.is_playing:
            self.is_playing = False
            self.midi.note_off(self.note, 127)



midi_player = init_pygame()

cap = cv.VideoCapture(0)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

linesNumber = 5
threshold = 0.2
hand_hitbox = frameWidth // 20

harp_strings = [HarpString(p0, p1, ndx, midi_player) for ndx, (p0, p1) in enumerate(get_line_points(frameWidth, frameHeight, linesNumber))]

for harp_string in harp_strings:
    harp_string.play_sound()
    time.sleep(1)
    harp_string.stop_sound()
    time.sleep(1)

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")


while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (320, 240), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]


    points = []
    for i in [4, 7]:

        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        if conf > threshold:
            points.append((int(x), int(y)))
            cv.circle(frame, (int(x), int(y)), hand_hitbox, (0, 0, 255))


    for harp_string in harp_strings:
        is_hit = any([(harp_string.distance(point) < hand_hitbox) if point else None for point in points])
        if is_hit:
            harp_string.play_sound()
        else:
            harp_string.stop_sound()

        cv.line(frame, harp_string.p0, harp_string.p1, (0, 0, 255) if is_hit else (0, 255, 0), 3)

    cv.imshow('Harp', frame)
