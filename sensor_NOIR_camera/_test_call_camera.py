#https://076923.github.io/posts/Python-opencv-2/
import cv2

# 카메라(인덱스) 호출
capture = cv2.VideoCapture(0)
# 카메라 너비 및 높이
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 키입력 대기함수(밀리초)
while cv2.waitKey(33) < 0:
    # 카메라 상태 저장 정상작동시 True 반환, 현재 시점 프레임 저장
    ret, frame = capture.read()
    # 윈도우 창의 제목과 이미지 할당
    cv2.imshow("VideoFrame", frame)
# 메모리 해제 메서드
capture.release()
# 모든 윈도우 창 제거 함수
cv2.destroyAllWindows("VideoFrame")

