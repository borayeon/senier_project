import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from picamera2 import Picamera2
import serial

# Serial port settings
ser = serial.Serial('/dev/ttyAMA4', baudrate=9600, timeout=1)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14]
BLINK_RESET_TIME = 60
# Thresholds and detection settings
EAR_THRESHOLD = 0.15  # Initial Eye Aspect Ratio threshold for drowsiness
YAWN_THRESHOLD = 20  # Yawn detection threshold (could be adjusted based on actual needs)
INITIAL_EYE_FRAME_THRESHOLD = 9
INITIAL_MOUTH_FRAME_THRESHOLD = 15
FRAME_RATE = 15  # Assuming 15 fps for dynamic calculations

# Global variable to store the latest received data
received_int = None


def serial_read():
    global received_int
    while True:
        data = ser.read(2)  # Read 2 bytes
        if data:
            received_int = int.from_bytes(data, byteorder='big')
            print("Received:", received_int)


# Start the serial reading in a separate thread
thread = threading.Thread(target=serial_read)
thread.daemon = True
thread.start()


class DrowsinessDetector:
    def __init__(self):
        self.start_time = time.time()
        self.eye_frame_count = 0
        self.mouth_frame_count = 0
        self.eye_frame_threshold = INITIAL_EYE_FRAME_THRESHOLD
        self.mouth_frame_threshold = INITIAL_MOUTH_FRAME_THRESHOLD
        self.blink_count = 0
        self.last_blink_timestamp = 0
        self.blinks_per_minute = 0

    def check_blink(self, ear):
        current_time = time.time()
        if ear < EAR_THRESHOLD and self.last_blink_timestamp == 0:
            self.last_blink_timestamp = current_time
        elif ear >= EAR_THRESHOLD and self.last_blink_timestamp != 0:
            self.blink_count += 1
            self.last_blink_timestamp = 0

        # Reset blink count every minute and update blinks per minute
        if current_time - self.start_time > BLINK_RESET_TIME:
            self.blinks_per_minute = self.blink_count
            self.blink_count = 0
            self.start_time = current_time
        return self.blinks_per_minute

    def update_thresholds(self, elapsed_time):
        # Dynamically adjust thresholds based on elapsed time
        self.eye_frame_threshold = max(INITIAL_EYE_FRAME_THRESHOLD - int(elapsed_time / 3600 * 3.5), 2)
        self.mouth_frame_threshold = max(INITIAL_MOUTH_FRAME_THRESHOLD - int(elapsed_time / 3600 * 2.5), 10)

    def check_drowsiness(self, ear, mor):
        # Update thresholds based on the current time
        elapsed_time = time.time() - self.start_time
        self.update_thresholds(elapsed_time)

        if ear < EAR_THRESHOLD:
            self.eye_frame_count += 1
        else:
            self.eye_frame_count = 0

        if mor > YAWN_THRESHOLD:
            self.mouth_frame_count += 1
        else:
            self.mouth_frame_count = 0

        # Check if the counts exceed dynamically adjusted thresholds
        if self.eye_frame_count >= self.eye_frame_threshold or self.mouth_frame_count >= self.mouth_frame_threshold:
            return True
        return False


detector = DrowsinessDetector()


def get_landmark_point(face_landmarks, landmark_index, image_shape):
    return np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [
        image_shape[1], image_shape[0]]


def eye_aspect_ratio(eye_points):
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    return (V1 + V2) / (2.0 * H)


def calculate_lip_distance(face_landmarks, image_shape):
    upper_lip_point = np.array(
        [face_landmarks.landmark[MOUTH_INDICES[0]].x, face_landmarks.landmark[MOUTH_INDICES[0]].y]) * [image_shape[1],
                                                                                                       image_shape[0]]
    lower_lip_point = np.array(
        [face_landmarks.landmark[MOUTH_INDICES[1]].x, face_landmarks.landmark[MOUTH_INDICES[1]].y]) * [image_shape[1],
                                                                                                       image_shape[0]]
    return np.linalg.norm(upper_lip_point - lower_lip_point)


def estimate_face_height(face_landmarks, image_shape):
    forehead_center = get_landmark_point(face_landmarks, 10, image_shape)
    chin_bottom = get_landmark_point(face_landmarks, 152, image_shape)
    return np.linalg.norm(forehead_center - chin_bottom)


def main():
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2.0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            image = picam2.capture_array()
            if image is None:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            yawn_status = "No Yawn Detected"
            sleep_status = "Awake"
            ear_text = "EAR: -"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lip_distance = calculate_lip_distance(face_landmarks, image.shape)
                    left_eye_points = [get_landmark_point(face_landmarks, index, image.shape) for index in
                                       LEFT_EYE_INDICES]
                    right_eye_points = [get_landmark_point(face_landmarks, index, image.shape) for index in
                                        RIGHT_EYE_INDICES]
                    ear = (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0
                    ear_text = f"EAR: {ear:.2f}"
                    blinks_per_minute = detector.check_blink(ear)
                    blink_text = f"Blinks/Min: {blinks_per_minute}"

                    if detector.check_drowsiness(ear, lip_distance):
                        yawn_status = "Drowsiness Detected!"
                        sleep_status = "Driver is Drowsy"

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

            flipped_image = cv2.flip(image, 1)
            cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Face Mesh', flipped_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    ser.close()
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
