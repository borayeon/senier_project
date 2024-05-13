# CO2, model, mp 연결 완료
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from picamera2 import Picamera2
import serial


# 0 : 좋음, 1 : 평범, 2 : 주의, 3 : 피로유발, 4 : 위험 - 태경
class CO2Sensor:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        self.receive_buff = [0] * 8
        self.level = 0  # 레벨 유지
        self.pre_level = 0
        self.weight = 0
    # 초기화 작업
    def setup(self):
        self.ser.flushInput()
        self.ser.flushOutput()
        time.sleep(2)

    # CO2 센서로 데이터 요청 명령 전송
    def send_cmd(self):
        send_data = [0x11, 0x01, 0x01, 0xED]
        for data in send_data:
            self.ser.write(bytes([data]))
            time.sleep(0.001)

    # 데이터의 체크섬 계산
    def checksum_cal(self):
        SUM = sum(self.receive_buff[:7])
        checksum = (256 - SUM % 256) % 256
        return checksum

    # ppm 값에 따라 CO2 상태 분류
    def co2_level(self, ppm):
        if ppm <= 1000:
            # 0단계
            return 0  # 좋음
        elif ppm <= 2000:
            # 1단계
            return 1  # 평범
        elif 1000 < ppm <= 2500:
            # 2단계
            return 2  # 주의 - 환기 or 외기순환 제안
        elif 2500 < ppm <= 3000:
            # 3단계
            return 3  # 피로 유발 - 환기 or 외기순환 제안
        else:
            # 4단계
            return 4  # 위험 - 환기 or 외기순환 제안

    def co2_weight(self):
        if self.level == 0:
            # 0단계
            self.level = 0
            return 0  
        elif self.level == 1:
            # 1단계
            return 0.3  
        elif self.level == 2:
            # 2단계
            return 0.5  
        elif self.level == 3:
            # 3단계
            return 0.8 
        elif self.level <= 4:
            # 4단계
            return 1.0

    def co2_check(self):
        self.ser.setup()
        print("Sending")
        self.send_cmd()
        time.sleep(1)

        recv_cnt = 0
        time.sleep(1)
        # 센서로부터 8바이트의 데이터 수신
        while recv_cnt < 8:
            if self.ser.in_waiting > 0:
                self.receive_buff[recv_cnt] = ord(self.ser.read(1))
                recv_cnt += 1
        # 수신된 데이터의 체크섬 검사
        if self.checksum_cal() == self.receive_buff[7]:
            PPM_Value = (self.receive_buff[3] << 8) | self.receive_buff[4]
            # 레벨 반환
            self.level = self.co2_level(PPM_Value)  # 0,1,2,3,4
            # 이전 레벨 < 현재 ppm -> 이전레벨 + 1
            if self.pre_level < self.level:
                self.level = self.pre_level + 1
            # 이전 레벨 > 현재 ppm -> 현재 ppm
            #elif self.pre_level > self.level:
            #    pass
                #self.level = self.level
            # 이전 레벨 == 현재 ppm -> 이전 레벨
            #elif self.pre_level == self.level:
            #    pass
                #self.level = self.pre_level
            # 가중치 반환
            self.weight = self.co2_weight()
            #print("PPM:", PPM_Value, "가중치:", self.weight)
            # 이전 레벨 저장
            self.pre_level = self.level
            return [self.level, self.weight]
        else:
            #print("CHECKSUM Error 이전 레벨로 입력됩니다.")
            return [self.level, self.weight]

    # CO2 상태를 지속적으로 확인하는 루프
    def run(self):
        time.sleep(900) # 15분 뒤부터 코드 동작
        while True:
            self.co2_check()
            time.sleep(600) # 10분마다 코드 동작

# Serial port settings
ser = serial.Serial('/dev/ttyAMA4', baudrate=9600, timeout=1)
sensor = CO2Sensor()

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils  # cv출력용
mp_drawing_styles = mp.solutions.drawing_styles  # cv출력용
mp_face_mesh = mp.solutions.face_mesh
# 얼굴의 관심 영역 인덱스 (눈, 입 등)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14]
# 눈 깜빡임, 하품 검출 기준값
BLINK_RESET_TIME = 60
EAR_THRESHOLD = 0.15  # 눈 깜빡임 기준값
YAWN_THRESHOLD = 20  # 하품 감지 기준값
# 초기 프레임 수
INITIAL_EYE_FRAME_THRESHOLD = 9
INITIAL_MOUTH_FRAME_THRESHOLD = 15
FRAME_RATE = 15  # FPS 기준값
# 졸음 상태 저장용 배열 (하품, 눈 깜빡임 지속 시간, 분당 눈 깜빡임 횟수, 분당 눈 감은 시간)
drowsy = [None, None, None, None]  # 하품(가중치 : 1 or 0), 눈 깜빡임 지속시간, 분당 눈 깜빡임 횟수, 분당 눈 감은 시간
previous_state = drowsy.copy()
# 시리얼 통신을 통한 데이터 수신
received_int = None

def serial_read():
    global received_int
    while True:
        # 2바이트 데이터 수신
        data = ser.read(2)
        if data:
            received_int = int.from_bytes(data, byteorder='big')
            print("Received:", received_int)

# 시리얼 데이터 수신 스레드 시작
thread = threading.Thread(target=serial_read)
thread2 = threading.Thread(target=sensor.run)
thread.daemon = True
thread2.daemon = True
thread.start()
thread2.start()

class DrowsinessDetector:
    def __init__(self):
        # 졸음 감지에 필요한 초기값 설정
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
        # 하품
        self.yawn_start_time = 0  # 시작시간
        self.yawn_duration = 0  # 기간
        self.yawn_count = 0  # 횟수
        self.stage = 0  # 상태
        self.stage_start_time = time.time()  # 시간
        self.weight_yawming = 0
        # 지속시간
        self.weight_eye_close_time = 0

    ## 하품 - 태경
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
                    if self.yawn_duration >= 2.5:  # 하품이 2.5초 이상 지속되었다면
                        self.yawn_count += 1
                        self.weight_yawming = self.yawning_update_stage()  # 0, 1
                        self.yawn_start_time = 0
                        self.yawn_duration = 0
                        return self.weight_yawming
        else:
            # 하품이 감지되지 않으면 시작 시간과 지속 시간 리셋
            self.yawn_start_time = 0
            self.yawn_duration = 0

        return self.weight_yawming
    # 하품 가중치
    def yawning_update_stage(self):
        current_time = time.time()  # 실시간
        elapsed_time = current_time - self.stage_start_time  ## 단계 시작 시간
        old_stage = self.stage

        if self.stage == 0 and self.yawn_count >= 1:
            self.stage = 1
        elif self.stage == 1 and elapsed_time >= 300:  # 1단계 : 5분간 지속
            if self.yawn_count == 0:  # 5분간 0회 시 0단계 격하
                self.stage = 0
            elif self.yawn_count == 1:  # 5분간 1회 시 1단계 유지
                self.stage = 1
            elif self.yawn_count == 2:  # 5분간 2회 시 2단계 격상
                self.stage = 2
            elif self.yawn_count >= 3:  # 5분간 3회 시 3단계 격상
                self.stage = 3
        elif self.stage == 2 and elapsed_time >= 900:  # 2단계 : 15분간 지속
            if self.yawn_count <= 1:
                self.stage = 1
            elif self.yawn_count == 2:
                self.stage = 2
            elif self.yawn_count >= 3:
                self.stage = 3
        elif self.stage == 3 and elapsed_time >= 1800:  # 3단계 : 30분간 지속
            if self.yawn_count <= 1:
                self.stage = 1
            elif self.yawn_count == 2:
                self.stage = 2
            elif self.yawn_count >= 3:
                self.stage = 3

        if old_stage != self.stage:  # 단계가 변경된 경우
            self.stage_start_time = current_time
            self.yawn_count = 0  # 단계가 변경되면 하품 횟수 초기화
            if self.stage != 0:
                return 1  # 단계가 0이 아닐 때 가중치 1 반환
            else:
                return 0  # 단계가 0일 때 0 반환
        else:  # 변경이 되지 않았다면
            if self.stage != 0:
                return 1  # 단계가 0이 아닐 때 가중치 1 반환
            else:
                return 0  # 단계가 0일 때 0 반환

    # 눈의 EAR 계산 함수 - 태경
    def eye_aspect_ratio(self, eye_points):
        V1 = np.linalg.norm(eye_points[1] - eye_points[5])
        V2 = np.linalg.norm(eye_points[2] - eye_points[4])
        H = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (V1 + V2) / (2.0 * H)
        return ear

    # EAR 계산을 통한 졸음 감지 - 태경
    def EAR_calculation(self, face_landmarks, image_shape):
        left_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in LEFT_EYE_INDICES])
        right_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in RIGHT_EYE_INDICES])
        left_ear = self.eye_aspect_ratio(left_eye_points)
        right_ear = self.eye_aspect_ratio(right_eye_points)

        min_ear = min(left_ear, right_ear)

        return min_ear

    # 500ms 이상 눈 감은 횟수로 졸음 확인 - 태경
    def check_drowsiness_(self):
        drowsy_count = len([d for d in self.closing_durations if d >= 0.5])
        weight = self.calculate_weight(drowsy_count)
        return weight

    def calculate_weight(self, drowsy_count):
        # 횟수에 따른 기본 가중치 설정
        if drowsy_count == 0:
            base_weight = 0
        elif drowsy_count == 1:
            base_weight = 0.1
        elif drowsy_count == 2:
            base_weight = 0.3
        elif drowsy_count >= 3:
            base_weight = 0.6

        # 횟수에 따른 추가 가중치 설정
        if drowsy_count < 3:
            additional_weight = 0
        elif drowsy_count < 5:
            additional_weight = 0.2
        else:
            additional_weight = 0.3

        return base_weight + additional_weight

    # 눈 지속 시간 계산 - 태경
    def calculate_eye_closing_time(self, ear):
        current_time = time.time()
        # 눈의 개방 정도(ear)가 임계값(EAR_THRESHOLD)보다 작고, 눈을 감기 시작한 시간이 기록되지 않았다면
        if ear < EAR_THRESHOLD and self.blink_timestamp == 0:
            self.blink_timestamp = current_time  # 현재 시간을 눈 감기 시작 시간으로 설정
            return 0  # 눈이 아직 완전히 감기지 않았음을 나타내는 0 반환

        # 눈의 개방 정도가 임계값 이상이고, 눈을 감기 시작한 시간이 기록되어 있다면 (즉, 눈이 다시 열렸다면)
        elif ear >= EAR_THRESHOLD and self.blink_timestamp != 0:
            duration = current_time - self.blink_timestamp  # 눈을 감고 있던 총 시간 계산
            if len(self.closing_durations) >= 15:
                self.closing_durations.pop(0)  # 리스트의 크기를 15으로 유지하기 위해 가장 오래된 기록 삭제
            self.closing_durations.append(duration)  # 새로운 눈 감김 지속 시간을 리스트에 추가
            self.blink_timestamp = 0  # 눈 감기 시작 시간 초기화
            self.blink_count += 1  # 눈 깜박임 카운트 증가
            return self.check_drowsiness_()  # 눈 깜박임이 3회 이상이면 true 반환

        return 0  # 눈 깜박임이 감지되지 않았다면 0 반환

    # 분당 눈 깜박임 횟수 계산 - 태경
    def calculate_blink_count_and_rate(self):
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

    # 눈 깜빡임 상태를 업데이트하는 함수 - 태경
    def update_eye_closure(self, ear):
        current_time = time.time()
        if ear < EAR_THRESHOLD:
            # 눈이 감기 시작했을 때 타임스탬프 기록
            if self.eye_closed_timestamp == 0:  # 눈이 감기 시작했을 때
                self.eye_closed_timestamp = current_time
        else:
            # 눈이 다시 열렸을 때 닫힌 눈의 시간을 계산하고 타임스탬프 초기화
            if self.eye_closed_timestamp != 0:  # 눈이 다시 열렸을 때
                self.closed_eye_time += current_time - self.eye_closed_timestamp
                self.eye_closed_timestamp = 0

    # PERCLOS (눈 감긴 상태 비율) 계산 함수 - 태경
    def calculate_perclos(self):
        current_time = time.time()
        # 60초마다 PERCLOS 계산
        if current_time - self.start_measurement_time >= 60:
            total_time = current_time - self.start_measurement_time
            perclos = (self.closed_eye_time / total_time) * 100
            # 측정 시간 및 닫힌 눈의 시간 리셋
            self.start_measurement_time = current_time
            self.closed_eye_time = 0
            # PERCLOS가 20% 미만인지 확인하여 상태 반환
            return perclos > 20
        return False  # 아직 60초가 지나지 않았다면 False 반환

    # 눈 깜빡임 횟수를 확인하는 함수 - 태경
    def check_blink(self, ear):
        current_time = time.time()
        # EAR 값이 임계값보다 작고 마지막 깜빡임 타임스탬프가 없을 때 깜빡임 시작
        if ear < EAR_THRESHOLD and self.last_blink_timestamp == 0:
            self.last_blink_timestamp = current_time
        # EAR 값이 임계값 이상이고 마지막 깜빡임 타임스탬프가 있을 때 깜빡임 종료
        elif ear >= EAR_THRESHOLD and self.last_blink_timestamp != 0:
            self.blink_count += 1
            self.last_blink_timestamp = 0

        # 1분마다 깜빡임 횟수를 초기화하고 갱신
        if current_time - self.start_time > BLINK_RESET_TIME:
            self.blinks_per_minute = self.blink_count
            self.blink_count = 0
            self.start_time = current_time
        return self.blinks_per_minute

    # 사용하지 않는? 코드
    # # 동재
    # # 동적으로 임계값을 조정하는 함수
    # def update_thresholds(self, elapsed_time):
    #     # Dynamically adjust thresholds based on elapsed time
    #     self.eye_frame_threshold = max(INITIAL_EYE_FRAME_THRESHOLD - int(elapsed_time / 3600 * 3.5), 2)
    #     self.mouth_frame_threshold = max(INITIAL_MOUTH_FRAME_THRESHOLD - int(elapsed_time / 3600 * 2.5), 10)
    #
    # # 졸음 여부를 확인하는 함수
    # def check_drowsiness(self, ear, mor):
    #     # 경과 시간에 따라 임계값 업데이트
    #     elapsed_time = time.time() - self.start_time
    #     self.update_thresholds(elapsed_time)
    #     # EAR 값에 따라 눈의 상태 카운트 증가 또는 초기화
    #     if ear < EAR_THRESHOLD:
    #         self.eye_frame_count += 1
    #     else:
    #         self.eye_frame_count = 0
    #     # 입 크기(mor)에 따라 입 상태 카운트 증가 또는 초기화
    #     if mor > YAWN_THRESHOLD:
    #         self.mouth_frame_count += 1
    #     else:
    #         self.mouth_frame_count = 0
    #
    #     # 눈과 입 카운트가 임계값을 초과하는지 확인하여 졸음 여부 반환
    #     if self.eye_frame_count >= self.eye_frame_threshold or self.mouth_frame_count >= self.mouth_frame_threshold:
    #         return True
    #     return False

# 졸음 감지기 인스턴스 생성 - 기본
detector = DrowsinessDetector()


# 얼굴 랜드마크의 좌표를 반환하는 함수 - 기본
def get_landmark_point(face_landmarks, landmark_index, image_shape):
    return np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [
        image_shape[1], image_shape[0]]

# 눈의 EAR를 계산하는 함수 - 태경
def eye_aspect_ratio(eye_points):
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    return (V1 + V2) / (2.0 * H)

# 입간격 확인 함수 및 얼굴 높이 추정 함수
# 입 간격을 계산하는 함수 - 태경
# def calculate_lip_distance(face_landmarks, image_shape):
#     upper_lip_point = np.array(
#         [face_landmarks.landmark[MOUTH_INDICES[0]].x, face_landmarks.landmark[MOUTH_INDICES[0]].y]) * [image_shape[1],
#                                                                                                        image_shape[0]]
#     lower_lip_point = np.array(
#         [face_landmarks.landmark[MOUTH_INDICES[1]].x, face_landmarks.landmark[MOUTH_INDICES[1]].y]) * [image_shape[1],
#                                                                                                        image_shape[0]]
#     return np.linalg.norm(upper_lip_point - lower_lip_point)


# 얼굴 높이를 추정하여 함수 - 태경
# def estimate_face_height(face_landmarks, image_shape):
#     forehead_center = get_landmark_point(face_landmarks, 10, image_shape)
#     chin_bottom = get_landmark_point(face_landmarks, 152, image_shape)
#     return np.linalg.norm(forehead_center - chin_bottom)

# 메인 함수: 카메라 영상 및 얼굴 메쉬 감지 루프

# 메인 함수: 카메라 영상 및 얼굴 메쉬 감지 루프
def main():
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2.0)
    print(drowsy)
    # MediaPipe 얼굴 메쉬 감지 설정
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            image = picam2.capture_array()
            if image is None:
                break
            # 이미지를 RGB로 변환 후, 얼굴 메쉬 감지
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # ----------------------------------------------------------------------------------------
                    # 하품
                    # 입 크기 계산 - 실시간 업데이트
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
            # 기본 상태 초기화
            #yawn_status = ""
            #sleep_status = ""
            #ear_text = ""
            flipped_image = cv2.flip(image, 1)
            #cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,cv2.LINE_AA)
            #cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,cv2.LINE_AA)
            #cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Face Mesh', flipped_image)
            print(drowsy)

            # 'Esc' 키 입력 시 루프 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # 센서 및 카메라 종료
    sensor.ser.close()
    ser.close()
    picam2.stop()
    cv2.destroyAllWindows()

# 메인 함수 실행
if __name__ == "__main__":
    main()
