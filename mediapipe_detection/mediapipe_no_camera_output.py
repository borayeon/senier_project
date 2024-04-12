# 필요한 라이브러리를 불러옵니다.
import cv2
import mediapipe as mp
import numpy as np
import time

# 얼굴 메시 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh

# 입과 눈 랜드마크 인덱스
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# 하품 감지를 위한 입 개방 임계값
YAWN_THRESHOLD = 20  # 이 값은 실험을 통해 조정할 수 있습니다.
# 졸음 감지를 위한 눈 개방 임계값
EAR_THRESHOLD = 0.21  # 이 값은 실험을 통해 조정할 수 있습니다.

# 눈 깜빡임 감지를 위한 변수 초기화
blink_timestamp = 0  # 눈이 마지막으로 감겼던 시간
eyes_open_time = 0  # 눈이 떠있던 시간
status = "평균"  # 사용자 상태 초기값

blink_count = 0
start_time = time.time()

# 입 크기 계산
def calculate_lip_distance(face_landmarks, image_shape):
    upper_lip_point = np.array([face_landmarks.landmark[UPPER_LIP].x, face_landmarks.landmark[UPPER_LIP].y]) * [
        image_shape[1], image_shape[0]]
    lower_lip_point = np.array([face_landmarks.landmark[LOWER_LIP].x, face_landmarks.landmark[LOWER_LIP].y]) * [
        image_shape[1], image_shape[0]]
    distance = np.linalg.norm(upper_lip_point - lower_lip_point)
    return distance
# 눈 크기 계산
def eye_aspect_ratio(eye_points):
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (V1 + V2) / (2.0 * H)
    return ear
def EAR_calculation(face_landmarks, image_shape):
    left_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [image_shape[1], image_shape[0]] for index in LEFT_EYE_INDICES])
    right_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [image_shape[1], image_shape[0]] for index in RIGHT_EYE_INDICES])
    return (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0

# 눈 지속 시간
def calculate_eye_open_time(ear, current_time):
    global blink_timestamp
    if ear < EAR_THRESHOLD and blink_timestamp == 0:
        blink_timestamp = current_time  # 눈 감김 시작
        return False
    elif ear >= EAR_THRESHOLD and blink_timestamp != 0:
        blink_timestamp = 0  # 눈 떠짐, 시간 초기화
        return True  # 눈 깜박임 감지
    return False
# 분당 눈 깜박임 횟수
def calculate_blink_rate(current_time):
    global blink_count, start_time
    elapsed_time = current_time - start_time
    if elapsed_time >= 60:
        blink_rate = blink_count / (elapsed_time / 60)  # 분당 깜박임 수
        blink_count = 0  # 카운터 리셋
        start_time = current_time  # 시간 리셋
        if 15 <= blink_rate <= 20:
            return "상태: 평균"
        elif blink_rate > 20 or blink_rate < 5:
            return "상태: 피곤"
    return "계산 중"
# 웹캠을 사용하기 위한 VideoCapture 객체를 생성합니다. 0은 기본 카메라를 의미합니다.
cap = cv2.VideoCapture(0)

# FaceMesh 모델을 사용하는 부분입니다.
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 이미지에서 얼굴 메시를 처리하고 상태를 분석하는 부분은 유지됩니다.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 입 크기 계산
                lip_distance = calculate_lip_distance(face_landmarks, image.shape)
                # ear 계산
                ear = EAR_calculation(face_landmarks)
                # 눈 지속 시간
                eot = calculate_eye_open_time(ear, time.time())
                # 눈 깜박임 빈도수
                # blinked = calculate_eye_open_time(ear, time.time())
                # if blinked:
                #     blink_count += 1  # 눈 깜박임 카운트 증가
                # status = calculate_blink_rate(time.time())
                # if status != "계산 중":
                #     print(status)
                # 분당 눈 감는 비율

                # 고개 숙임

                #

                # 상태 처리 로직 (예: 하품 감지, 졸음 상태 감지 등)은 여기에 포함됩니다.
                # 하지만 결과를 화면에 표시하는 부분은 제거됩니다.

        # ESC 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 사용이 끝난 후 카메라를 해제합니다.
cap.release()
