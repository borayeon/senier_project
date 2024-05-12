# CO2, model, mp 연결 완료
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from picamera2 import Picamera2
import serial

# co2 데이터 수집
received_co2 = None


# 0 : 좋음, 1 : 평범, 2 : 주의, 3 : 피로유발, 4 : 위험
class CO2Sensor:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        self.receive_buff = [0] * 8

    def setup(self):
        self.ser.flushInput()
        self.ser.flushOutput()
        time.sleep(2)

    def send_cmd(self):
        send_data = [0x11, 0x01, 0x01, 0xED]
        for data in send_data:
            self.ser.write(bytes([data]))
            time.sleep(0.001)

    def checksum_cal(self):
        SUM = sum(self.receive_buff[:7])
        checksum = (256 - SUM % 256) % 256
        return checksum

    def co2_level(self, ppm):
        if ppm <= 400:
            return 0  # 좋음
        elif ppm <= 1000:
            return 1  # 평범
        elif 1000 < ppm <= 2000:
            return 2  # 주의
        elif 2000 < ppm <= 5000:
            return 3  # 피로 유발
        else:
            return 4  # 위험

    def co2_check(self):
        global received_co2
        self.setup()
        print("Sending...")
        self.send_cmd()
        time.sleep(1)

        recv_cnt = 0
        while recv_cnt < 8:
            if self.ser.in_waiting > 0:
                self.receive_buff[recv_cnt] = ord(self.ser.read(1))
                recv_cnt += 1

        if self.checksum_cal() == self.receive_buff[7]:
            PPM_Value = (self.receive_buff[3] << 8) | self.receive_buff[4]
            received_co2 = self.co2_level(PPM_Value)
            print("PPM:", PPM_Value, " 상태:", received_co2)
            return received_co2
        else:
            print("CHECKSUM Error")
            received_co2 = -1
            return received_co2
        time.sleep(1)

    def run(self):
        while True:
            self.co2_check()

# Serial port settings
ser = serial.Serial('/dev/ttyAMA4', baudrate=9600, timeout=1)
sensor = CO2Sensor()

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils  # cv출력용
mp_drawing_styles = mp.solutions.drawing_styles  # cv출력용
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
drowsy = [False, False, False, False]  # 하품, 눈 깜빡임 지속시간, 분당 눈 깜빡임 횟수, 분당 눈 감은 시간
previous_state = drowsy.copy()
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
thread2 = threading.Thread(target=sensor.run)
thread.daemon = True
thread2.daemon = True
thread.start()
thread2.start()


class DrowsinessDetector:
    def __init__(self):
        self.blink_count = 0
        self.start_time = time.time()
        self.blink_timestamp = 0
        self.closing_durations = []
        self.eye_frame_count = 0
        self.mouth_frame_count = 0
        self.eye_frame_threshold = INITIAL_EYE_FRAME_THRESHOLD
        self.mouth_frame_threshold = INITIAL_MOUTH_FRAME_THRESHOLD
        self.last_blink_timestamp = 0
        self.blinks_per_minute = 0
        self.closed_eye_time = 0
        self.start_measurement_time = time.time()
        self.eye_closed_timestamp = 0

    ## 하품
    def detect_yawning(self, face_landmarks, image_shape):
        # 입 크기 계산
        upper_lip_point = np.array(
            [face_landmarks.landmark[MOUTH_INDICES[0]].x, face_landmarks.landmark[MOUTH_INDICES[0]].y]) * [
                              image_shape[1], image_shape[0]]
        lower_lip_point = np.array(
            [face_landmarks.landmark[MOUTH_INDICES[1]].x, face_landmarks.landmark[MOUTH_INDICES[1]].y]) * [
                              image_shape[1], image_shape[0]]
        lip_distance = np.linalg.norm(upper_lip_point - lower_lip_point)
        # print(lip_distance)
        # 하품 감지 및 지속 시간 처리
        if lip_distance > YAWN_THRESHOLD:
            if self.yawn_start_time == 0:  # 하품이 시작된 시간이 기록되지 않았다면
                self.yawn_start_time = time.time()  # 현재 시간을 하품 시작 시간으로 기록
            else:
                self.yawn_duration = time.time() - self.yawn_start_time  # 하품 지속 시간 계산
                if self.yawn_duration >= 2.5:  # 하품이 3초 이상 지속되었다면
                    # text_status.drowsy0_status = "Yawn Detected"
                    return True
        else:
            self.yawn_start_time = 0  # 하품이 감지되지 않으면 시작 시간을 리셋
            self.yawn_duration = 0  # 지속 시간도 리셋
            # text_status.drowsy0_status = "No Yawn Detected"
            return False
        return False

    # EAR 계산 코드_1
    def eye_aspect_ratio(self, eye_points):
        V1 = np.linalg.norm(eye_points[1] - eye_points[5])
        V2 = np.linalg.norm(eye_points[2] - eye_points[4])
        H = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (V1 + V2) / (2.0 * H)
        return ear

    # EAR 계산 코드_2
    def EAR_calculation(self, face_landmarks, image_shape):
        left_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in LEFT_EYE_INDICES])
        right_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in RIGHT_EYE_INDICES])
        left_ear = self.eye_aspect_ratio(left_eye_points)
        right_ear = self.eye_aspect_ratio(right_eye_points)

        min_ear = min(left_ear, right_ear)

        return min_ear

    # 500ms 이상 눈 감은 횟수가 3회 이상인지 확인
    def check_drowsiness_(self):
        return len([d for d in self.closing_durations if d >= 0.5]) >= 3

    # 눈 지속 시간 계산
    def calculate_eye_closing_time(self, ear):
        # global blink_timestamp, closing_durations, blink_count
        current_time = time.time()
        # 눈의 개방 정도(ear)가 임계값(EAR_THRESHOLD)보다 작고, 눈을 감기 시작한 시간이 기록되지 않았다면
        if ear < EAR_THRESHOLD and self.blink_timestamp == 0:
            self.blink_timestamp = current_time  # 현재 시간을 눈 감기 시작 시간으로 설정
            return False  # 눈이 아직 완전히 감기지 않았음을 나타내는 False 반환

        # 눈의 개방 정도가 임계값 이상이고, 눈을 감기 시작한 시간이 기록되어 있다면 (즉, 눈이 다시 열렸다면)
        elif ear >= EAR_THRESHOLD and self.blink_timestamp != 0:
            duration = current_time - self.blink_timestamp  # 눈을 감고 있던 총 시간 계산
            if len(self.closing_durations) >= 10:
                self.closing_durations.pop(0)  # 리스트의 크기를 10으로 유지하기 위해 가장 오래된 기록 삭제
            self.closing_durations.append(duration)  # 새로운 눈 감김 지속 시간을 리스트에 추가
            self.blink_timestamp = 0  # 눈 감기 시작 시간 초기화
            self.blink_count += 1  # 눈 깜박임 카운트 증가
            return self.check_drowsiness_()  # 눈 깜박임이 3회 이상이면 true 반환

        return False  # 눈 깜박임이 감지되지 않았다면 False 반환

    # 분당 눈 깜박임 횟수 계산
    def calculate_blink_count_and_rate(self):
        # global blink_count, start_time
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            blink_rate = self.blink_count  # 분당 깜박임 수
            self.blink_count = 0  # 카운터 리셋
            self.start_time = current_time  # 시간 리셋
            if 15 <= blink_rate <= 20:  # 안 피곤
                return False
            elif blink_rate < 5 or blink_rate > 20:  # 피곤
                return True
        return False

    # perclos 20% 계산
    def update_eye_closure(self, ear):
        # global eye_closed_timestamp, closed_eye_time, start_measurement_time, total_time
        current_time = time.time()
        if ear < EAR_THRESHOLD:
            if self.eye_closed_timestamp == 0:  # 눈이 감기 시작했을 때
                self.eye_closed_timestamp = current_time
        else:
            if self.eye_closed_timestamp != 0:  # 눈이 다시 열렸을 때
                self.closed_eye_time += current_time - self.eye_closed_timestamp
                self.eye_closed_timestamp = 0

    def calculate_perclos(self):
        # global start_measurement_time, total_time, closed_eye_time
        current_time = time.time()
        if current_time - self.start_measurement_time >= 60:  # 60초마다 PERCLOS 계산
            total_time = current_time - self.start_measurement_time
            perclos = (self.closed_eye_time / total_time) * 100
            self.start_measurement_time = current_time  # 측정 시간 리셋
            self.closed_eye_time = 0  # 닫힌 눈의 시간 리셋
            if perclos < 20:
                return True  # PERCLOS가 20% 미만인지 확인
            else:
                return False
        return False  # 아직 60초가 지나지 않았다면 False 반환

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
    print(drowsy)
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
                    # ----------------------------------------------------------------------------------------
                    # # 하품
                    # # 입 크기 계산 - 실시간 업데이트
                    drowsy[0] = detector.detect_yawning(face_landmarks, image.shape)
                    # ----------------------------------------------------------------------------------------
                    # 눈 깜빡임 지속시간 - 실시간 업데이트
                    # ear 계산
                    ear = detector.EAR_calculation(face_landmarks, image.shape)
                    # 눈 깜빡임 시 눈 감김 지속 시간이 500ms을 넘으면 상태 True로 변환
                    drowsy[1] = detector.calculate_eye_closing_time(ear)
                    # ----------------------------------------------------------------------------------------
                    # 눈 깜박임 분당 빈도수 - 1분마다 업데이트
                    drowsy[2] = detector.calculate_blink_count_and_rate()
                    # ----------------------------------------------------------------------------------------
                    # perclos
                    # 눈 상태 업데이트
                    detector.update_eye_closure(ear)  # 현재 EAR 값과 현재 시간 전달
                    # 1분마다 업데이트
                    drowsy[3] = detector.calculate_perclos()
                    # ----------------------------------------------------------------------------------------
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            # cv 출력용
            flipped_image = cv2.flip(image, 1)
            cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Face Mesh', flipped_image)
            print(drowsy)
            # if drowsy != previous_state:
            #    print(drowsy)
            #    previous_state=drowsy.copy()
            if cv2.waitKey(5) & 0xFF == 27:
                break

    sensor.close()  # co2 센서 닫기
    ser.close()
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
