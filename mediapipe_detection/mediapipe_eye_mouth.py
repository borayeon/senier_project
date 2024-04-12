# 필요한 라이브러리를 불러옵니다.
import cv2
import mediapipe as mp
import numpy as np

# 그리기 유틸리티와 얼굴 메시 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 입과 눈 랜드마크 인덱스
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# 하품 감지를 위한 입 개방 임계값
YAWN_THRESHOLD = 20  # 이 값은 실험을 통해 조정할 수 있습니다.
# 졸음 감지를 위한 눈 개방 임계값
EAR_THRESHOLD  = 0.21  # 이 값은 실험을 통해 조정할 수 있습니다.



# 입 크기 계산
def calculate_lip_distance(face_landmarks, image_shape):
    upper_lip_point = np.array([face_landmarks.landmark[UPPER_LIP].x, face_landmarks.landmark[UPPER_LIP].y]) * [image_shape[1], image_shape[0]]
    lower_lip_point = np.array([face_landmarks.landmark[LOWER_LIP].x, face_landmarks.landmark[LOWER_LIP].y]) * [image_shape[1], image_shape[0]]
    distance = np.linalg.norm(upper_lip_point - lower_lip_point)
    return distance

# 눈 크기 계산
def eye_aspect_ratio(eye_points):
    # 세로 거리 계산
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # 가로 거리 계산
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    # EAR 계산
    ear = (V1 + V2) / (2.0 * H)
    return ear

# 랜드마크 좌표를 실제 이미지 상의 좌표로 변환합니다.
def get_landmark_point(face_landmarks, landmark_index, image_shape):
    landmark_point = np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [image_shape[1], image_shape[0]]
    return landmark_point

# 얼굴 메시 그리기 설정을 정의합니다. 선의 두께와 원의 반지름을 설정합니다.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 웹캠을 사용하기 위한 VideoCapture 객체를 생성합니다. 0은 기본 카메라를 의미합니다.
cap = cv2.VideoCapture(0)

# FaceMesh 모델을 사용하는 부분입니다. 최대 얼굴 수, 랜드마크의 세부사항, 감지 및 추적의 신뢰도 설정이 포함됩니다.
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    # 웹캠이 열려있는 동안 반복합니다.
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

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
        # 얼굴 메시 결과가 있으면, 각 얼굴에 대해 메시를 그립니다.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
            # 상태 인식부
                # 눈 랜드마크 포인트 추출 및 EAR 계산
                left_eye_points = np.array(
                    [get_landmark_point(face_landmarks, index, image.shape) for index in LEFT_EYE_INDICES])
                right_eye_points = np.array(
                    [get_landmark_point(face_landmarks, index, image.shape) for index in RIGHT_EYE_INDICES])
                left_EAR = eye_aspect_ratio(left_eye_points)
                right_EAR = eye_aspect_ratio(right_eye_points)
                ear = (left_EAR + right_EAR) / 2.0
                ear_text = f"EAR: {ear:.2f}"  # EAR 값을 문자열로 변환
                # 하품 거리
                lip_distance = calculate_lip_distance(face_landmarks, image.shape)
            # 상태 처리부
                # 하품 감지
                if lip_distance > YAWN_THRESHOLD:
                    yawn_status = "Yawn Detected!"
                # EAR 기반 졸음 감지
                if ear > EAR_THRESHOLD:
                    sleep_status = "Awake"
                else:
                   sleep_status = "Sleepy"
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
                    .DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
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
        cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            cv2.LINE_AA)
        # 졸음 감지 상태 표시
        cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # 졸음 감지 상태 표시
        cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Face Mesh', flipped_image)

    # ESC 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 사용이 끝난 후 카메라를 해제합니다.
cap.release()
