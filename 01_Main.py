import threading
from Facial_Utils import *
from Facenet_Architecture import *
import os
import cv2
import imutils
import time
from datetime import datetime
import math
import mediapipe as mp
import numpy as np
import pathlib

"""################## FACIAL GLOBAL SETTING ##################"""
# Set-up path
WEIGHT = "F:/PROJECT/Selfie_Drone/Yolo/yolov3-wider_16000.weights"
MODEL = "F:/PROJECT/Selfie_Drone/Yolo/yolov3-face.cfg"
# Set-up net
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_SIZE = 416
CROP_SIZE = 160  # required
# BGR
BLUE = (200, 70, 45)  # Blue
GREEN = (30, 180, 40)  # Green
RED = (10, 10, 200)  # Red
# Facenet
model_facenet = InceptionResNetV1()
# KNN
K_UNKWOWN_THRESHOLD = 9
K_NB = 5
MODEL_NAME = 'KNN.h5'
# Init
thread_finished = True
total_save = 0
boxes_buffer = []
text_buffer = []
# Area threshold to switch fly-mode
arae_track = [27000, 37000]

"""################## FACIAL FUNCTIONS ##################"""
print("Mode: Predict")
predict_mode = True

# Load photos from folder 'data' and feed to KNN
trainX, trainy = load_dataset('F:/PROJECT/Selfie_Drone/Controller_data')
model_facenet.load_weights('F:/PROJECT/Selfie_Drone/facenet_keras_weights.h5')
newTrainX, trainy, out_encoder = convert_dataset(model_facenet, trainX, trainy)
knn_model = KNN_fit(newTrainX, MODEL_NAME, K_NB)

print(trainX.shape, trainy.shape)
print(newTrainX.shape)

"""############################################################"""
# Set-up center of camera, PID for PID controller.
center = [320, 240]
# Set-up key-world.
UP = "UP"
DOWN = "DOWN"
CW = "TURN RIGHT"
CCW = "TURN LEFT"
forward = ["RIGHT-ARM CLOSE", "FORWARD"]
backward = ["RIGHT-ARM OPEN", "BACKWARD"]
left = ["LEFT-ARM CLOSE", "MOVE LEFT"]
right = ["LEFT-ARM OPEN", "MOVE RIGHT"]
KeepDistance = ["TWO HANDS UP", "KEEP-DISTANCE"]
TakePicture = ["TWO HAND CROSS", "Take-picture"]
ApproachLand = ["LEFT HAND FORWARD", "Approach and landing"]
# Set-up picture folder
Picture_Folder = "F:/PROJECT/Selfie_Drone/Picture"

''' ############################ FACIAL RECOGNITION, GET INFOR ########################### '''
def detect_and_predict(net, frame, IMG_SIZE, predict_mode, model_facenet):
    global thread_finished, boxes_buffer, text_buffer
    thread_finished = False

    boxes_buffer = []
    text_buffer = []

    boxes_buffer, confidences = face_detection(net, frame, IMG_SIZE)

    for box in boxes_buffer:
        # Extract position data
        topleft_x, topleft_y, width, height = box[0], box[1], box[2], box[3]

        # Extract face area
        face = frame[topleft_y:topleft_y + height, topleft_x:topleft_x + width]

        if predict_mode and (face.size > 0):
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
            # Extract features
            face_emb = get_embedding(model_facenet, resized_face)
            face_emb_array = np.asarray(face_emb)
            face_emb_array = face_emb_array.reshape(1, -1)
            # Predict
            text = KNN_predict(knn_model, face_emb_array, trainy, out_encoder, K_UNKWOWN_THRESHOLD)
        else:
            text = f'{confidences[0]:.2f}'

        # BUFFER
        text_buffer.append(text)

    thread_finished = True
    return thread_finished


def face_infor(frame, thread_finished, boxes_buffer, text_buffer):
    global boxes_cache, text_cache
    if thread_finished:
        # Save result
        boxes_cache = boxes_buffer.copy()
        text_cache = text_buffer.copy()
        x = threading.Thread(target=detect_and_predict,
                             args=(net, frame, IMG_SIZE, predict_mode, model_facenet,)).start()
    area = 0
    cx, cy = 320, 240
    for i, box in enumerate(boxes_cache):
        x, y, w, h = box[:4]
        margin_x, margin_y = 5, 10
        topleft = (x - margin_x, y - margin_y)
        bottomright = (x + w + margin_x, y + h + margin_y)
        cv2.rectangle(frame, topleft, bottomright, BLUE, 2)
        cv2.putText(frame, text_cache[i], topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        if text_cache[i][2: 6] == "Vinh":
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            cv2.circle(frame, (cx, cy), 3, GREEN, 2)
        else:
            area = 0
    return frame, [cx, cy, area]


''' ############################ CALCULATE TIME PROCESS ########################### '''
def Counting_Time(time_start, time_waiting, condition_1=True, condition_2=True):
    time_process = 0
    time_now = datetime.now()
    if condition_1 is True:
        time_waiting = (time_now - time_start).seconds
        '''now condition_1 much be turn from True to False at this point to lock waiting time'''
    if condition_2 is True:
        time_pass = (time_now - time_start).seconds
        time_process = time_pass - time_waiting
    return time_process, time_waiting


''' ############################ MAIN-CLASS ==> CREATE POSE-LANDMARK ########################### '''
class HolisticDetector:

    def __init__(self, mode=False, Complexity=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.Complexity = Complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_Draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(self.mode, self.Complexity, self.smooth,
                                                  self.detectionCon, self.trackCon)

    def find_holistic(self, img):
        self.result = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.result:
            return img

    def find_pose_landmarks(self, img, draw=True):
        pose_lm = []
        if self.result.pose_landmarks:
            self.mp_Draw.draw_landmarks(img, self.result.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                        self.mp_Draw.DrawingSpec(color=(225, 0, 255), circle_radius=5),
                                        self.mp_Draw.DrawingSpec(color=(140, 189, 219)))
            for index, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                pose_lm.append([cx, cy, index])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 100, 100), cv2.FILLED)
                    cv2.putText(img, str(index), (cx, int(cy + 20)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (36, 255, 12), 2, cv2.LINE_AA)
        return img, pose_lm


''' ############################################ POSE DETECT ############################################ '''
''' forward, backward, left, right, take picture, keep distance'''
# Angle between two point by y-axis.
def vertical_angle(a, b):
    if a is None or b is None:
        return None
    else:
        return math.degrees(math.atan2(a[0] - b[0], a[1] - b[1]))


# Detect Pose command.
def pose_detect(pose_lm, pose=None):
    if pose_lm:
        noise = pose_lm[0]
        L_shoulder = pose_lm[11]
        R_shoulder = pose_lm[12]
        L_elbow = pose_lm[13]
        R_elbow = pose_lm[14]
        L_wrist = pose_lm[15]
        R_wrist = pose_lm[16]

        right_arm_angle = vertical_angle(R_elbow, R_wrist)
        left_arm_angle = vertical_angle(L_elbow, L_wrist)
        if L_wrist[1] < L_shoulder[1]:
            if left_arm_angle > 30:
                pose = left
            elif left_arm_angle < -30:
                pose = right
            if R_wrist[1] < R_shoulder[1]:
                if L_wrist[0] > R_wrist[0] and L_wrist[1] > noise[1]:
                    pose = KeepDistance
        elif L_wrist[1] > L_shoulder[1] and R_wrist[1] > R_shoulder[1]:
            if (R_shoulder[0] - L_wrist[0]) < (L_shoulder[0] - L_wrist[0]) and \
                    (L_shoulder[0] - R_wrist[0]) > (R_shoulder[0] - L_wrist[0]) and \
                    L_wrist[0] < R_wrist[0]:
                if L_elbow[1] > L_wrist[1] and R_elbow[1] > R_wrist[1] and \
                        40 < left_arm_angle:
                    pose = TakePicture

        if R_wrist[1] < R_shoulder[1]:
            if right_arm_angle < -30:
                pose = forward
            elif right_arm_angle > 30:
                pose = backward
        if (L_wrist[1] - L_shoulder[1]) < (L_elbow[1] - L_wrist[1]) and \
                L_shoulder[0] - 3 < L_wrist[0] < L_shoulder[0] + 3 and \
                left_arm_angle < 20:
            pose = ApproachLand
    return pose


''' ########################################## POSE - COMMAND ############################################ '''
def Pose_Command(pose, distance, threshold_distance, PITCH, ROW=0):
    if pose is not None:
        if distance < threshold_distance:
            if pose == forward:
                PITCH = 0
        else:
            if pose == forward:
                PITCH = 35
            if pose == backward:
                PITCH = -35
            if pose == left:
                ROW = -35
            if pose == right:
                ROW = 35
    return PITCH, ROW


''' ####################################### DISTANCE - ARROW-LINE ####################################### '''
# Shoulder Width = 33cm
actual_shoulderWidth = 33
# Pixel Width *0.026 to convert to cm
picWidth = 350 * 0.026
knowDistance = 76
# Calculate camera focalLength in centimeter.
camera_focalLength = (knowDistance * picWidth) / actual_shoulderWidth
# Calculate distance from object to camera in centimeter.
def Two_point_Distance(p1, p2):
    Distance = 0
    if p1 and p2:
        p2[0] = np.clip(p2[0], 0, 640)
        p2[1] = np.clip(p2[1], 0, 480)
        ''' Pixel * 0.026 to convert to cm '''
        Distance = math.sqrt((int(p1[0]) - int(p2[0])) ** 2 + (int(p1[1]) - int(p2[1])) ** 2) * 0.026
    return int((33 * camera_focalLength) / Distance)


# Draw arrow from camera center to object point
def Draw_Arrow(img, p1, p2):
    p2[0] = np.clip(p2[0], 0, 640)
    p2[1] = np.clip(p2[1], 0, 480)
    return cv2.arrowedLine(img, (p1[0], p1[1]), (p2[0], p2[1]), (222, 100, 25), thickness=3)


''' ########################################## AUTO - TRACKING ########################################## '''
def Throttle_Tracking(cy, p2, pid, throttle_Perror):
    if p2:
        error = cy - p2[1]
        speed = pid[0] * error + pid[2] * (error - throttle_Perror)
        speed = np.clip(speed, -100, 100)
        if speed < 0:
            pose = DOWN
        else:
            pose = UP
    else:
        speed, error, pose = [0, 0, None]
    return speed, error, pose


def Yaw_Tracking(cx, p2, pid, yaw_Perror):
    if p2:
        error = p2[0] - cx
        speed = pid[0] * error + pid[2] * (error - yaw_Perror)
        speed = np.clip(speed, -100, 100)
        if speed < 0:
            pose = CCW
        else:
            pose = CW
    else:
        speed, error, pose = [0, 0, None]
    return speed, error, pose


''' ########################################     Keep-Distance    ####################################### '''
def Track_Distance(distance, pose, pre_pose, distance_track, is_tracking, pid, distance_Perror, threshold_distance):
    if pre_pose == KeepDistance and pre_pose != pose:
        if is_tracking is False:
            distance_track = distance
            is_tracking = True
        else:
            distance_track = None
            is_tracking = False

    if distance and distance_track is not None:
        if pose == ApproachLand:
            error = (distance - threshold_distance)
        else:
            error = (distance - distance_track)
        speed = pid[0] * error + pid[2] * (error - distance_Perror)
        speed = np.clip(speed, -100, 100)
        if speed < 0:
            pose = backward
        else:
            pose = forward
    else:
        speed, error, pose = [0, 0, None]
    return distance_track, is_tracking, int(speed), error, pose


''' ########################################     Take-picture    ####################################### '''
''' if use check variable to set time-waiting before take picture  '''
def Take_picture(frame, picture, pose, pre_pose, time_waiting, time_start, number_next, check):
    is_take_picture = False
    if pre_pose == TakePicture and pre_pose != pose:
        is_take_picture = True
        check = True

    time_process, time_waiting = Counting_Time(time_start, time_waiting,
                                               condition_1=is_take_picture is True,
                                               condition_2=True)
    if time_process == 1:
        cv2.putText(frame, "Take Picture in: 3", (10, 210), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 2:
        cv2.putText(frame, "Take Picture in: 2", (10, 210), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 3:
        cv2.putText(frame, "Take Picture in: 1", (10, 210), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 3 and check is True:
        picture = np.copy(picture)
        picture = imutils.resize(picture, width=1200)
        file_name = "picture_" + str(number_next) + ".jpg"
        cv2.imwrite(os.path.join(Picture_Folder, file_name), picture)
        check = False
    return frame, time_waiting, check


def Picture_Path():
    Picture_Folder_Path = pathlib.Path(Picture_Folder)
    number_max = -1
    number_next = 0
    picture_show = None
    for item in Picture_Folder_Path.glob('*'):
        if item.is_file():
            if number_max < int(item.name.split("_")[-1].split('.')[0]):
                number_max = int(item.name.split("_")[-1].split('.')[0])
                picture_show = item.name
                number_next = number_max + 1
    Path = f"C:/Users/PC/PycharmProjects/Final_Project/Picture/{picture_show}"
    return Path, number_next


''' ##########################################     PUT TEXT    ########################################## '''
def Stt(img, pretext, value, x, y, threshold_distance=None, font=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
        color=GREEN, thickness=2, lineType=cv2.LINE_AA):
    if value is not None:
        if value > 0:
            if y == 180 and (value < threshold_distance):
                color = RED
            else:
                color = BLUE
        elif value < 0:
            color = RED
        elif value == 0:
            color = BLUE
        return cv2.putText(img, f'{pretext}: {abs(int(value))}', (x, y), font,
                           fontScale, color, thickness, lineType)
    else:
        return cv2.putText(img, pretext, (x, y), font,
                           fontScale, color, thickness, lineType)


def Stt_Condition(img, pose, is_tracking, value, x, y):
    if y == 120:
        if pose == forward or pose == backward:
            return Stt(img, pose[1], value, x, y)
        elif is_tracking is True or pose == ApproachLand:
            if value >= 0:
                return Stt(img, forward[1], value, x, y)
            elif value < 0:
                return Stt(img, backward[1], value, x, y)
        else:
            return Stt(img, forward[1], 0, x, y)
    if y == 150:
        if pose == left or pose == right:
            return Stt(img, pose[1], value, x, y)
        else:
            return Stt(img, right[1], 0, x, y)
    if y == 210 and is_tracking is True:
        return Stt(img, KeepDistance[1], value, x, y)
    if y == 240 and pose:
        if is_tracking is True:
            return Stt(img, f"POSE: {pose[0]}", None, x, y)
        else:
            return Stt(img, f"POSE: {pose[0]}", None, x, y - 30)


''' #########################################    LANDING   ######################################## '''
def is_Landing(frame, Landing, pose, distance, threshold_distance, time_start, time_waiting):
    time_process, time_waiting = Counting_Time(time_start, time_waiting,
                                               condition_1=pose == ApproachLand and
                                                           (threshold_distance + 10 < distance) and
                                                           (distance < threshold_distance + 17),
                                               condition_2=True)
    if time_process == 1:
        cv2.putText(frame, "LANDING IN: 3", (10, 240), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 2:
        cv2.putText(frame, "LANDING IN: 2", (10, 240), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 3:
        cv2.putText(frame, "LANDING IN: 1", (10, 240), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
    if time_process == 4:
        Landing = True
    return frame, Landing, time_waiting


''' #######################################     TURN ON OPTION    ####################################### '''
def Turn_On_Option(frame, cover_time, cover, time_waiting, time_Start):
    threshold_color = 5
    option = None
    TakeOff = False
    frame_color = frame.mean()
    if frame_color < threshold_color:
        if cover is False:
            cover_time += 1
            cover = True
    else:
        cover = False

    time_process, waiting_time, = Counting_Time(time_Start, time_waiting,
                                                condition_1=cover_time == 1 and cover is True,
                                                condition_2=cover_time > 0)
    if time_process >= 4:
        if cover_time == 3:
            option = "mode 1"
        elif cover_time == 4:
            option = "mode 2"

    if option is None:
        cv2.putText(frame, f'CODE: {cover_time}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, BLUE, 2, cv2.LINE_AA)

    if option is not None:
        if time_process == 4:
            cv2.putText(frame, "TAKE_OFF IN 3s ", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
        if time_process == 5:
            cv2.putText(frame, "TAKE_OFF IN 2s ", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
        if time_process == 6:
            cv2.putText(frame, "TAKE_OFF IN 1s ", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2, cv2.LINE_AA)
        if time_process >= 7:
            TakeOff = True

    return TakeOff, option, cover_time, cover, waiting_time, time_process


''' #####################################################################################################
    ##########################################       MAIN      ########################################## '''
def main():
    Time_now = datetime.now()

    cover = False
    cover_time = 0
    TakeOff_Waiting_Time = 0

    TakeOff = False
    option = None

    Perror_Throttle = 0
    Perror_Yaw = 0
    Perror_distance = 0
    PID = [0.4, 0, 0.4]

    pose = None
    pre_pose = None
    DISTANCE_Track = None
    is_tracking = False

    Pic_Waiting_Time = 0
    check = False

    threshold_distance = 100
    Landing = True
    Landing_Waiting_Time = 0

    cap = cv2.VideoCapture(0)
    pass_time = 0
    detector = HolisticDetector()
    while True:
        success, frame_0 = cap.read()
        frame = frame_0.copy()
        # Trun-on fly-mode
        if Landing is True:
            TakeOff, option, cover_time, cover, TakeOff_Waiting_Time, TakeOff_time_process = Turn_On_Option(frame,
                                                                                                            cover_time,
                                                                                                            cover,
                                                                                                            TakeOff_Waiting_Time,
                                                                                                            Time_now)
            if option is not None and TakeOff_time_process >= 7:
                Landing = False
                cover_time = 0
                pose = None
        # MODE_1: Pose-Command
        if TakeOff is True and option == "mode 1" and Landing == False:
            frame = detector.find_holistic(frame)
            frame, pose_lm = detector.find_pose_landmarks(frame, draw=False)
            # Start post detect
            # Stop post detect When Drone approach to land
            if pose != ApproachLand:
                pre_pose = pose
                pose = pose_detect(pose_lm)
            if pose_lm:
                Draw_Arrow(frame, center, pose_lm[0])
                ''' Distance between Drone and controller '''
                DISTANCE = Two_point_Distance(pose_lm[11], pose_lm[12])
                ''' Check if tracking mode on '''
                DISTANCE_Track, is_tracking, PITCH, Perror_distance, pose_Pitch = Track_Distance(DISTANCE, pose,
                                                                                                 pre_pose,
                                                                                                 DISTANCE_Track,
                                                                                                 is_tracking,
                                                                                                 PID, Perror_distance,
                                                                                                 threshold_distance)
                ''' Auto keep latitude and angle (THROTTLE and YAW) with value'''
                THROTTLE, Perror_Throttle, pose_Throttle = Throttle_Tracking(center[1], pose_lm[0],
                                                                             PID, Perror_Throttle)
                YAW, Perror_Yaw, pose_Yaw = Yaw_Tracking(center[0], pose_lm[0], PID, Perror_Yaw)
                ''' Pose-Command: move forward/backward (PITCH), move left/right (ROW) with value '''
                PITCH, ROW = Pose_Command(pose, DISTANCE, threshold_distance, PITCH)
                ''' Pose-Command: Take picture '''
                path, number_next = Picture_Path()
                frame, Pic_Waiting_Time, check = Take_picture(frame, frame_0, pose, pre_pose,
                                                              Pic_Waiting_Time, Time_now,
                                                              number_next, check)
                ''' Pose-Command: Approach and landing '''
                frame, Landing, Landing_Waiting_Time = is_Landing(frame, Landing, pose, DISTANCE,
                                                                  threshold_distance, Time_now,
                                                                  Landing_Waiting_Time)

                ''' Print status: Up/down, trun-left/turn-right, move-left/move-right, DISTANCE value'''
                Stt(frame, pose_Throttle, THROTTLE, 10, 60)
                Stt(frame, pose_Yaw, YAW, 10, 90)
                Stt_Condition(frame, pose, is_tracking, PITCH, 10, 120)
                Stt_Condition(frame, pose, is_tracking, ROW, 10, 150)
                Stt(frame, "DISTANCE", DISTANCE, 10, 180, threshold_distance)
                Stt_Condition(frame, pose, is_tracking, DISTANCE_Track, 10, 210)
                ''' Print status: Pose command '''
                Stt_Condition(frame, pose, is_tracking, None, 10, 240)
                ''' Print status: THROTTLE, YAW, PITCH, ROW, DISTANCE value '''
                print(int(THROTTLE), int(YAW), int(PITCH), ROW)
                Stt(frame, f'T:{int(THROTTLE)}', None, 100, 450, color=BLUE)
                Stt(frame, f'Y:{int(YAW)}', None, 240, 450, color=BLUE)
                Stt(frame, f'P:{int(PITCH)}', None, 350, 450, color=BLUE)
                Stt(frame, f'R:{int(ROW)}', None, 480, 450, color=BLUE)
        if Landing is True:
            option = None

        # MODE_2: facial detect (detect controller only)
        if TakeOff is True and option == "mode 2":
            ''' Get infor: [cx, cy, area] '''
            frame, infor = face_infor(frame, thread_finished, boxes_buffer, text_buffer)
            ''' Draw_arrow '''
            infor[0] = np.clip(infor[0], 0, 640)
            infor[1] = np.clip(infor[1], 0, 480)
            Draw_Arrow(frame, center, infor[0:2])
            ''' Auto keep latitude, angle and move forward to the controller '''
            THROTTLE, Perror_Throttle, pose_Throttle = Throttle_Tracking(center[1], infor[0:2],
                                                                         PID, Perror_Throttle)
            Stt(frame, pose_Throttle, THROTTLE, 10, 60)
            YAW, Perror_Yaw, pose_Yaw = Yaw_Tracking(center[0], infor[0:2], PID, Perror_Yaw)
            Stt(frame, pose_Yaw, YAW, 10, 90)

            if arae_track[0] < infor[2] < arae_track[1]:
                mode_2_PITCH = 0
                Stt(frame, forward[1], mode_2_PITCH, 10, 120)
            elif infor[2] < arae_track[0]:
                mode_2_PITCH = 25
                Stt(frame, forward[1], mode_2_PITCH, 10, 120)
            elif infor[2] > arae_track[1]:
                mode_2_PITCH = - 25
                Stt(frame, backward[1], mode_2_PITCH, 10, 120)

            if arae_track[0] < infor[2] < arae_track[1]:
                option = 'mode 1'
                infor[2] = None

        ''' Print FPS '''
        current_time = time.time()
        fps = 1 / (current_time - pass_time)
        pass_time = current_time
        Stt(frame, "FPS", fps, 10, 30)
        frame = imutils.resize(frame, width=800)
        cv2.imshow("stream", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
