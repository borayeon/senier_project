#https://076923.github.io/posts/Python-opencv-10/

import cv2
# 이미지 읽기
src = cv2.imread("Image/crow.jpg", cv2.IMREAD_COLOR)
# 이미지를 색상 변환 코드로 변환
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
