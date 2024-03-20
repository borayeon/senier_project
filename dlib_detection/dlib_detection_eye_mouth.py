import numpy as np
import dlib
import cv2
import time

# Dlib의 얼굴 랜드마크 모델을 사용하여 눈과 입의 위치를 찾는 데 사용할 인덱스
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))
MOUTH = list(range(48, 68))  # 입 주위 랜드마크

# 프레임 크기와 이름 설정
frame_width = 640
frame_height = 480
title_name = 'Drowsiness Detection'

# Haar cascade 파일 로드
face_cascade_name = '../data_file/opencv-4.x/opencv-4.x/data/haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

# Dlib 얼굴 랜드마크 예측기 로드
predictor_file = '../data_file/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)

# 변수 설정
status = 'Awake' # 상태
number_closed = 0 # 눈 감은 횟수
min_EAR = 0.20 # 졸음 판단 기준
closed_limit = 15  # -- 눈 감김이 N번 이상일 경우 졸음으로 간주
min_MAR = 25  # 하품으로 판단할 입의 최소 개방 크기, 실험을 통해 적절한 값을 찾아야 합니다.
yawn_limit = 3  # 연속적으로 하품이 감지되어야 하는 횟수
number_yawns = 0  # 하품 횟수를 카운트
yawning_detected = False
show_frame = None # 처리된 프레임 저장?
sign = None # 졸음 상태 나타내는 문자열 저장
color = None # 사용자 상태 판단 색상 저장

# EAR 계산 함수
def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

# YAR 계산 함수
def getMAR(points):
    # MAR(Mouth Aspect Ratio)을 계산하기 위한 포인트: 입의 상단(합: 61, 63, 65)과 하단(합: 67, 65, 63)의 거리
    A = np.linalg.norm(points[61] - points[63]) if len(points) > 61 else 0  # 상단
    B = np.linalg.norm(points[63] - points[65]) if len(points) > 65 else 0  # 중간
    C = np.linalg.norm(points[67] - points[65]) if len(points) > 67 else 0  # 하단
    MAR = (A + B + C) / 3.0
    return MAR


# 얼굴과 눈을 감지하고 표시하는 함수
def detectAndDisplay(image):
    global number_closed, color, show_frame,sign, status, number_yawns
    image = cv2.resize(image, (frame_width, frame_height))
    # 이미지 저장
    show_frame = image
    # 이미지 색상 변환하여 히스토그램 균일하게
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    # 얼굴이 감지되지 않았을 때 status 초기화
    if len(faces) == 0:
        status = 'Unknown'

    for (x, y, w, h) in faces:
        # 얼굴 직사각형 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 얼굴 랜드마크 찾기
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        show_part_E = points[EYES]
        show_part_M = points[MOUTH]

        # 눈 감지
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        # 졸음 평균
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2
        # 입 감지
        mouth_EAR = getMAR(points[MOUTH])

        right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0, 0], right_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0, 0], left_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 눈 포인트
        for (i, point) in enumerate(show_part_E):
            x = point[0, 0]
            y = point[0, 1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        # 입 포인트
        for (i, point) in enumerate(show_part_M):
            x = point[0, 0]
            y = point[0, 1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        # awake, sleep 상태 설정
        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if (number_closed < 0):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed += 1

        # 하품 감지 로직
        if mouth_EAR > min_MAR:
            number_yawns += 1
            if number_yawns >= yawn_limit:
                # 화면에 "Yawning detected!" 표시
                cv2.putText(show_frame, "Yawning detected!", (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
        else:
            number_yawns = 0  # 하품이 감지되지 않으면 카운터를 리셋합니다.

        sign = 'Sleep count: ' + str(number_closed) + ' / ' + str(closed_limit)

        # 졸음 확정시 알람 설정
        #if (number_closed > closed_limit):
        #    show_frame = frame_gray
            # winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME) #-- 본인 환경에 맞게 변경할 것

        # 얼굴이 검출된 경우에만 status와 sign을 표시
        cv2.putText(show_frame, status, (x - w, y - h), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
        cv2.putText(show_frame, sign, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow(title_name, show_frame)

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened:
    print('Could not open video')
    exit(0)

while True:
    ret, frame = cap.read()

    if frame is None:
        print('Could not read frame')
        cap.release()
        break

    detectAndDisplay(frame)

    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

