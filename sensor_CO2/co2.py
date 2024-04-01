import serial
import time

# 시리얼 통신 포트와 속도 설정
ser = serial.Serial('/dev/ttyAMA0', 9600)

# 초기화
def setup():
    # 시리얼 통신 초기화
    ser.flushInput()
    ser.flushOutput()
    time.sleep(2)

# 전송
def send_cmd():
    send_data = [0x11, 0x01, 0x01, 0xED]
    for data in send_data:
        ser.write(bytes([data]))
        time.sleep(0.001)


# 무결성 검증
def checksum_cal():
    # HEAD, LEN, CMD 및 모든 DATA 바이트의 합계를 구합니다.
    SUM = sum(Receive_Buff[:7])
    # 합계를 256으로 나눈 나머지를 구하고, 이를 256에서 빼서 체크썸을 계산합니다.
    checksum = (256 - SUM % 256) % 256
    return checksum


def co2_level(ppm):
    if  ppm <= 400:
        return 0  # 좋음
    elif ppm <= 1000:
        return 1 # 평범
    elif 1000 < ppm <= 2000:
        return 2 # 주의(졸음 유발)
    elif 2000 < ppm <= 5000:
        return 3 # 피로 유발(집중력 저하)
    else:
        return 4 # 위험
# 출처: https://www.co2meter.com/blogs/news/carbon-dioxide-indoor-levels-chart


# 반복
def loop():
    setup()
    print("Sending...")
    send_cmd()
    time.sleep(1)

    # 데이터 수신 - 8바이트 대기
    recv_cnt = 0
    while recv_cnt < 8:
        if ser.in_waiting > 0:
            Receive_Buff[recv_cnt] = ord(ser.read(1))
            recv_cnt += 1
    # 무결성 검증
    if checksum_cal() == Receive_Buff[7]:
        PPM_Value = (Receive_Buff[3] << 8) | Receive_Buff[4]
        status = co2_level(PPM_Value)
        print("PPM:", PPM_Value," 상태:", status)
    else:
        print("CHECKSUM Error")

    time.sleep(1)


if __name__ == "__main__":
    Receive_Buff = [0] * 8
    while True:
        loop()