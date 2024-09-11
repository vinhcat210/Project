import cv2
import mediapipe as mp
import numpy as np
import check_right_action
from check_right_action import get_theta, check_right_angles 
import time

if __name__ == "__main__":

    # Initialize MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    dict = {"Case_1": ['hand_raise.jpg'], "Case_2": ['hand_bend.jpg']}
    dict["Case_1"].append(check_right_action.return_right_data(dict, "Case_1"))
    dict["Case_2"].append(check_right_action.return_right_data(dict, "Case_2"))
    hand_action_angle = "Unknown"
    result = "Unknown"
    prev_y_elbow = None

    # Setup video capture
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = check_right_action.pre_processing(frame, pose)

            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass
            if results.pose_landmarks:
                l_s, r_s, l_e, l_w = check_right_action.get_left_arm_coordinate(landmarks) #ham lay toa do tay trai va phai

                # Vẽ đoạn nối giữa các điểm tay trái
                connections = [(11, 13), (13, 15)]  # Các kết nối giữa các điểm mốc
                for start_idx, end_idx in connections:
                    start_point = (
                        int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
                    end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

                    # Calculate the angle at the elbow
                    angle_b = check_right_action.calculate_angle_2d(l_s, l_e, l_w)
                    
                    hand_action_y = "Unknown"
                    hand_action_angle, case = check_right_action.determine_hand_action_based_on_angle(angle_b)   
                    
                    if prev_y_elbow is not None:
                        hand_action_y = check_right_action.determine_hand_action_based_on_y(l_e[1], prev_y_elbow)      
                    prev_y_elbow = l_e[1]
                    
                    if  case in dict:
                        alpha = get_theta(dict[case][1][0], dict[case][1][2], l_s, l_e)
                        beta = get_theta(dict[case][1][2], dict[case][1][3], l_e, l_w)
                        result = check_right_angles(alpha, beta)
                        
                    cv2.putText(image, f"Result: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA)     
                    cv2.putText(image, f"Angle Action: {hand_action_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Y Action: {hand_action_y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
        # Display the resulting frame
            cv2.imshow("Mediapipe Feed", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()








