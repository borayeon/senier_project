import cv2

def detect_faces_and_eyes():
    # 웹캠을 켭니다.
    cap = cv2.VideoCapture(0)

    # 얼굴 및 눈 검출을 위해 분류기를 로드합니다.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        # 웹캠으로부터 영상을 받아옵니다.
        ret, frame = cap.read()

        if not ret:
            break

        # 영상을 회색으로 변환합니다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴을 검출합니다.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 검출된 얼굴 주위에 사각형을 그립니다.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 얼굴 영역에서 눈을 검출합니다.
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # 검출된 눈 주위에 사각형을 그립니다.
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # 결과 영상을 화면에 출력합니다.
        cv2.imshow('Face and Eye Detection', frame)

        # 'q'를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠을 끕니다.
    cap.release()
    cv2.destroyAllWindows()

# 웹캠으로부터 얼굴과 눈을 인식합니다.
detect_faces_and_eyes()
