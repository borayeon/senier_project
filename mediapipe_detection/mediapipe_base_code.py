# 필요한 라이브러리를 불러옵니다.
import cv2
import mediapipe as mp

# MediaPipe의 그리기 유틸리티와 얼굴 메시 솔루션을 초기화합니다.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 얼굴 메시 그리기 설정을 정의합니다. 선의 두께와 원의 반지름을 설정합니다.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 웹캠을 사용하기 위한 VideoCapture 객체를 생성합니다. 0은 기본 카메라를 의미합니다.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
i = 0
# FaceMesh 모델을 사용하는 부분입니다. 최대 얼굴 수, 랜드마크의 세부사항, 감지 및 추적의 신뢰도 설정이 포함됩니다.
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    # 웹캠이 열려있는 동안 반복합니다.
    while cap.isOpened():
        print(i)
        i+=1
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            continue

        # 성능 최적화를 위해 이미지 쓰기를 비활성화합니다.
        image.flags.writeable = False
        # OpenCV의 BGR 이미지를 RGB로 변환합니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지에서 얼굴 메시를 처리합니다.
        results = face_mesh.process(image)

        # 이미지를 다시 쓰기 가능하게 하고, RGB에서 BGR로 다시 변환합니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 얼굴 메시 결과가 있으면, 각 얼굴에 대해 메시를 그립니다.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴의 메시, 윤곽, 눈동자 등을 그립니다.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # 화면에 보이는 이미지를 사용자가 더 쉽게 인식할 수 있도록 좌우 반전합니다.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        # ESC 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 사용이 끝난 후 카메라를 해제합니다.
cap.release()
