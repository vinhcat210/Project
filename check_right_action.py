import cv2
import mediapipe as mp
import numpy as np
import math

def pre_processing(frame, pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def get_data_from_image(image, landmarks, arm_indices, connections):
    for idx, landmark in enumerate(landmarks):
        if idx in arm_indices:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (245, 117, 66), -1)

        # Vẽ các kết nối cánh tay
    for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]

        # Tính tọa độ của điểm đầu và điểm cuối
        start_point = (int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0]))
        end_point = (int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0]))

        # Vẽ đường kết nối giữa hai điểm
        cv2.line(image, start_point, end_point, (245, 66, 230), 2)

def get_left_arm_coordinate(landmarks):
    arm_indices = [11, 13, 15, 12]
    x = [landmarks[idx].x for idx in arm_indices]
    y = [landmarks[idx].y for idx in arm_indices]
    z = [landmarks[idx].z for idx in arm_indices]

    left_shoulder = np.array([x[0], y[0]])
    right_shoulder = np.array([x[3], y[3]])
    left_elbow = np.array([x[1], y[1]])
    left_wrist = np.array([x[2], y[2]])

    return left_shoulder, right_shoulder, left_elbow, left_wrist


def return_right_data(dict, case):
    path = dict[case][0]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.9) as pose:
        results = pose.process(image)
        landmarks = results.pose_landmarks.landmark
        if results.pose_landmarks:
            right_coordinate_data = get_left_arm_coordinate(landmarks)
            return right_coordinate_data

def get_theta(q_1, q_2, v_1, v_2):
    q21 = q_2 - q_1
    v21 = v_2 - v_1
    sp = np.dot(q21, v21)
    theta = np.arccos(sp /((np.linalg.norm(q21))*(np.linalg.norm(v21))))
    return math.degrees(theta)



def check_right_angles(a, b):
    if a <= 15 and b <= 10:
        return 'right action'
    else:
        return 'wrong action'
    
# Function to calculate the 2D angle between three points with range from 0 to 180 degrees
def calculate_angle_2d(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    # Vectors from point b to a and from point b to c in 2D
    ba = a - b
    bc = c - b
    
    # Calculate the dot product and magnitudes
    dot_prod = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(dot_prod / (mag_ba * mag_bc))
    
    # Convert the angle to degrees
    angle = np.degrees(angle_radians)
    
    # Ensure the angle is within [0, 180] degrees
    if angle > 180:
        angle = 360 - angle
    
    return angle

# Function to determine hand action based on the angle
def determine_hand_action_based_on_angle(angle_b):
    if angle_b < 90:
        return "Bending", "Case_2"  # Hand is bending
    else:
        return "Extending", "Case_1"  # Hand is extending

# Function to determine hand action based on elbow y-coordinate change
def determine_hand_action_based_on_y(current_y, prev_y):
    if current_y > prev_y:
        return "Lowered"  # Hand is raisedq
    elif current_y < prev_y:
        return "Raised"  # Hand is lowered
    else:
        return ""

    