# 필요한 라이브러리를 불러옵니다.
import cv2
import mediapipe as mp
import numpy as np
import time

# 1
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#
# 얼굴 메시 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh

# 졸음 상태 변수
class Status():
    def __init__(self):
        self.BlinkingDuration = False
        self.BlinkingFrequency = False
        self.Yawning = False
        self.HeadDown = False
status = Status()

# 얼굴 랜드마크 인덱스 변수
class LandmarkIndex():
    def __init__(self):
        # 입과 눈 랜드마크 인덱스
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
landmark = LandmarkIndex()

# 임계치 변수
class Threshold():
    def __init__(self):
        # 하품 감지를 위한 입 개방 임계값
        self.YAWN_THRESHOLD = 20
        # 졸음 감지를 위한 눈 개방 임계값
        self.EAR_THRESHOLD = 0.21
        # 졸음 감지를 위한 고개숙임 임계값
        self.HEAD_THRESHOLD = 50
        # 고개숙임
        self.head_distance = 0
threshold = Threshold()


YAWN_THRESHOLD = 20
# 졸음 감지를 위한 눈 개방 임계값
EAR_THRESHOLD = 0.21
# 졸음 감지를 위한 고개숙임 임계값
HEAD_THRESHOLD = 50
# 고개숙임
head_distance = 0

# 전역 변수 초기화
blink_count = 5
start_time = time.time()


# 눈 깜박임 감지를 위한 변수
blink_timestamp = 0
closing_durations = []  # 최근 10번의 눈 감김 지속 시간



# perclos
total_time = 0
closed_eye_time = 0
start_measurement_time = time.time()
eye_closed_timestamp = 0

# 하품 시작 시간을 추적하기 위한 전역 변수
yawn_start_time = 0
yawn_duration = 0

drowsy_1 = False
drowsy_2 = False
drowsy_3 = False
drowsy_4 = False

#perclos = False
#head_down_count = False

### 개선 코드 작성 구간 1 - 하품
def detect_yawning(face_landmarks, image_shape, yawn_start_time):
    import numpy as np
    import time

    # 상입술과 하입술의 랜드마크 인덱스
    UPPER_LIP = landmark.UPPER_LIP
    LOWER_LIP = landmark.LOWER_LIP

    yawn_threshold = threshold.YAWN_THRESHOLD

    # 입 크기 계산
    upper_lip_point = np.array([face_landmarks.landmark[UPPER_LIP].x, face_landmarks.landmark[UPPER_LIP].y]) * [image_shape[1], image_shape[0]]
    lower_lip_point = np.array([face_landmarks.landmark[LOWER_LIP].x, face_landmarks.landmark[LOWER_LIP].y]) * [image_shape[1], image_shape[0]]
    lip_distance = np.linalg.norm(upper_lip_point - lower_lip_point)

    # 하품 상태
    yawn_status = "No Yawn Detected"
    yawn_duration = 0

    # 하품 감지 및 지속 시간 처리
    if lip_distance > yawn_threshold:
        if yawn_start_time == 0:  # 하품이 시작된 시간이 기록되지 않았다면
            yawn_start_time = time.time()  # 현재 시간을 하품 시작 시간으로 기록
        else:
            yawn_duration = time.time() - yawn_start_time  # 하품 지속 시간 계산
            if yawn_duration >= 2:  # 하품이 3초 이상 지속되었다면
                yawn_status = "Yawn Detected!"
    else:
        yawn_start_time = 0  # 하품이 감지되지 않으면 시작 시간을 리셋
        yawn_duration = 0  # 지속 시간도 리셋

    return yawn_status , yawn_duration, yawn_start_time
# 입 크기 계산
# def calculate_lip_distance(face_landmarks, image_shape):
#     upper_lip_point = np.array([face_landmarks.landmark[landmark.UPPER_LIP].x, face_landmarks.landmark[landmark.UPPER_LIP].y]) * [
#         image_shape[1], image_shape[0]]
#     lower_lip_point = np.array([face_landmarks.landmark[landmark.LOWER_LIP].x, face_landmarks.landmark[landmark.LOWER_LIP].y]) * [
#         image_shape[1], image_shape[0]]
#     distance = np.linalg.norm(upper_lip_point - lower_lip_point)
#     return distance
###
### 개선 코드 작성 구간 2 - 눈 깜빡임 지속시간
# 눈 크기 계산

#
def eye_aspect_ratio(eye_points):
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (V1 + V2) / (2.0 * H)
    return ear

def EAR_calculation(face_landmarks, image_shape):
    left_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [image_shape[1], image_shape[0]] for index in landmark.LEFT_EYE_INDICES])
    right_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [image_shape[1], image_shape[0]] for index in landmark.RIGHT_EYE_INDICES])
    return (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0


# 500ms 이상 눈 감은 횟수가 3회 이상인지 확인
def check_drowsiness():
    return len([d for d in closing_durations if d >= 0.5]) >= 3
# 눈 지속 시간 계산
def calculate_eye_closing_time(ear, current_time):
    global blink_timestamp, closing_durations, blink_count
    # 눈의 개방 정도(ear)가 임계값(EAR_THRESHOLD)보다 작고, 눈을 감기 시작한 시간이 기록되지 않았다면
    if ear < EAR_THRESHOLD and blink_timestamp == 0:
        blink_timestamp = current_time  # 현재 시간을 눈 감기 시작 시간으로 설정
        return False  # 눈이 아직 완전히 감기지 않았음을 나타내는 False 반환

    # 눈의 개방 정도가 임계값 이상이고, 눈을 감기 시작한 시간이 기록되어 있다면 (즉, 눈이 다시 열렸다면)
    elif ear >= EAR_THRESHOLD and blink_timestamp != 0:
        duration = current_time - blink_timestamp  # 눈을 감고 있던 총 시간 계산
        if len(closing_durations) >= 10:
            closing_durations.pop(0)  # 리스트의 크기를 10으로 유지하기 위해 가장 오래된 기록 삭제
        closing_durations.append(duration)  # 새로운 눈 감김 지속 시간을 리스트에 추가
        blink_timestamp = 0  # 눈 감기 시작 시간 초기화
        blink_count += 1 # 눈 깜박임 카운트 증가
        return check_drowsiness()  # 눈 깜박임이 3회 이상이면 true 반환

    return False  # 눈 깜박임이 감지되지 않았다면 False 반환

# 분당 눈 깜박임 횟수 계산
def calculate_blink_count_and_rate():
    global blink_count, start_time, drowsy_2
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= 60:
        blink_rate = blink_count / 60  # 분당 깜박임 수
        blink_count = 0  # 카운터 리셋
        start_time = current_time  # 시간 리셋
        if 15 <= blink_rate <= 20:
            drowsy_2 = False
            return False
        elif blink_rate < 5 or blink_rate > 20:
            drowsy_2 = True
            return True
        else:
            drowsy_2 = False
            return False

    return False
# perclos 20% 계산
def update_eye_closure(ear):
    global eye_closed_timestamp, closed_eye_time, start_measurement_time, total_time
    current_time = time.time()
    if ear < EAR_THRESHOLD:
        if eye_closed_timestamp == 0:  # 눈이 감기 시작했을 때
            eye_closed_timestamp = current_time
    else:
        if eye_closed_timestamp != 0:  # 눈이 다시 열렸을 때
            closed_eye_time += current_time - eye_closed_timestamp
            eye_closed_timestamp = 0

def calculate_perclos():
    global start_measurement_time, total_time, closed_eye_time, drowsy_3
    current_time = time.time()
    if current_time - start_measurement_time >= 5:  # 60초마다 PERCLOS 계산
        total_time = current_time - start_measurement_time
        perclos = (closed_eye_time / total_time) * 100
        start_measurement_time = current_time  # 측정 시간 리셋
        closed_eye_time = 0  # 닫힌 눈의 시간 리셋
        if perclos < 20:
            drowsy_3 = True  # PERCLOS가 20% 미만인지 확인
        else:
            drowsy_3 = False
    return False  # 아직 60초가 지나지 않았다면 False 반환

# 고개 숙임
def detect_head_down(face_landmarks, image_shape):
    # 코와 턱 랜드마크 인덱스
    nose_index = 5
    chin_index = 152

    # 코와 턱 랜드마크 좌표 추출
    nose_point = np.array([face_landmarks.landmark[nose_index].x, face_landmarks.landmark[nose_index].y]) * [image_shape[1], image_shape[0]]
    chin_point = np.array([face_landmarks.landmark[chin_index].x, face_landmarks.landmark[chin_index].y]) * [image_shape[1], image_shape[0]]

    # 코와 턱 사이의 거리 계산
    head_distance = np.linalg.norm(chin_point - nose_point)

    # 고개 숙임 여부 판단
    if head_distance > HEAD_THRESHOLD:
        return True
    else:
        return False
# 2
# 랜드마크 좌표를 실제 이미지 상의 좌표로 변환합니다.
def get_landmark_point(face_landmarks, landmark_index, image_shape):
    landmark_point = np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [image_shape[1], image_shape[0]]
    return landmark_point
# 얼굴 메시 그리기 설정을 정의합니다. 선의 두께와 원의 반지름을 설정합니다.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#

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

# 3
        # 성능 최적화를 위해 이미지 쓰기를 비활성화합니다.
        image.flags.writeable = False
        # OpenCV의 BGR 이미지를 RGB로 변환합니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지에서 얼굴 메시를 처리합니다.
        results = face_mesh.process(image)
        # 이미지를 다시 쓰기 가능하게 하고, RGB에서 BGR로 다시 변환합니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 상태
        yawn_status = "No Yawn Detected"  # 기본 상태는 '하품이 감지되지 않음'
        sleep_status = "Awake"  # 기본 상태
        ear_text = "EAR: -"
        drowsy_status = "drowsy_status"
        status_status = "status_status"
        perclos_status = "Perclos Detected"
        head_status = "head Detected"

        # 이미지에서 얼굴 메시를 처리하고 상태를 분석하는 부분은 유지됩니다.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
#----------------------------------------------------------------------------------------
                # # 하품
                # # 입 크기 계산 - 실시간 업데이트
                yawn_status, yawn_start_time, yawn_duration = detect_yawning(face_landmarks, image.shape, yawn_start_time)
#----------------------------------------------------------------------------------------
                # 눈 깜빡임 지속시간 - 실시간 업데이트
                # ear 계산
                ear = EAR_calculation(face_landmarks, image.shape)
                # 눈 깜빡임 시 눈 감김 지속 시간이 500ms을 넘으면 상태 True로 변환
                drowsy_1 = calculate_eye_closing_time(ear, time.time())
#----------------------------------------------------------------------------------------
                # 눈 깜박임 분당 빈도수 - 1분마다 업데이트
                _ = calculate_blink_count_and_rate()
#----------------------------------------------------------------------------------------
                # perclos
                # 눈 상태 업데이트
                update_eye_closure(ear)  # 현재 EAR 값과 현재 시간 전달
                # 1분마다 업데이트
                _ = calculate_perclos()
#----------------------------------------------------------------------------------------
                # 고개 숙임
                #head_down_count = detect_head_down(face_landmarks, image.shape)

                # 상태 처리 로직 (예: 하품 감지, 졸음 상태 감지 등)은 여기에 포함됩니다.
                # 하지만 결과를 화면에 표시하는 부분은 제거됩니다.
# 4
                # 상태 처리부
                # 상태 업데이트 로직

                # EAR 기반 졸음 감지
                if ear > EAR_THRESHOLD:
                    sleep_status = "Awake"
                else:
                    sleep_status = "Sleepy"

                # 눈 감김 지속시간
                if drowsy:
                    drowsy_status = "ect drowsy"
                else:
                    drowsy_status = "ect not drowsy"

                # 분당 눈 깜빡이
                if status:
                    status_status = "bun on"
                else:
                    status_status = "bun off"

                # perclos
                if perclos:
                    perclos_status = "perclos on"
                else:
                    perclos_status = "perclos off"

                # head
                if head_down_count:
                    head_status = "head on"
                else:
                    head_status = "head off"

                # 이미지 처리부
                # 얼굴의 메시, 윤곽, 눈동자 등을 그립니다.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                # 랜드마크 및 메시 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing
                    .DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_iris_connections_style())
        # 텍스트 및 이미지 출력부
        # 이미지를 좌우 반전합니다.
        flipped_image = cv2.flip(image, 1)
        # 하품 감지 상태를 좌우 반전된 영상에 표시
        cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        # 눈 크기
        cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                    cv2.LINE_AA)
        # 졸음 감지 상태 표시
        #cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # 눈 지속시간
        cv2.putText(flipped_image, drowsy_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # 분당 눈 깜빡임
        cv2.putText(flipped_image, status_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # perclos
        cv2.putText(flipped_image, perclos_status, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # head
        cv2.putText(flipped_image, str(head_distance), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face Mesh', flipped_image)
#
        # ESC 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 사용이 끝난 후 카메라를 해제합니다.
cap.release()
