import serial
import time

class CO2Sensor:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        self.receive_buff = [0] * 8

    def setup(self):
        self.ser.flushInput()
        self.ser.flushOutput()
        time.sleep(2)

    def send_cmd(self):
        send_data = [0x11, 0x01, 0x01, 0xED]
        for data in send_data:
            self.ser.write(bytes([data]))
            time.sleep(0.001)

    def checksum_cal(self):
        SUM = sum(self.receive_buff[:7])
        checksum = (256 - SUM % 256) % 256
        return checksum

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
    def loop(self):
        self.setup()
        print("Sending...")
        self.send_cmd()
        time.sleep(1)

        recv_cnt = 0
        while recv_cnt < 8:
            if self.ser.in_waiting > 0:
                self.receive_buff[recv_cnt] = ord(self.ser.read(1))
                recv_cnt += 1

        if self.checksum_cal() == self.receive_buff[7]:
            PPM_Value = (self.receive_buff[3] << 8) | self.receive_buff[4]
            status = self.co2_level(PPM_Value)
            print("PPM:", PPM_Value, " 상태:", status)
        else:
            print("CHECKSUM Error")

        time.sleep(1)

    def run(self):
#        while True:
       self.loop()# 1번 실행하기로 변경

if __name__ == "__main__":
    sensor = CO2Sensor()
    sensor.run()


from co2 import CO2Sensor

sensor = CO2Sensor()
sensor.run()

#
# import serial
# import time
#
# # 시리얼 통신 포트와 속도 설정
# ser = serial.Serial('/dev/ttyAMA0', 9600)
#
#
# def send_cmd():
#     send_data = [0x11, 0x01, 0x01, 0xED]
#     for data in send_data:
#         ser.write(bytes([data]))
#         time.sleep(0.001)
#
#
# def checksum_cal():
#     # HEAD, LEN, CMD 및 모든 DATA 바이트의 합계를 구합니다.
#     SUM = sum(Receive_Buff[:7])
#     # 합계를 256으로 나눈 나머지를 구하고, 이를 256에서 빼서 체크썸을 계산합니다.
#     checksum = (256 - SUM % 256) % 256
#     return checksum
#
#
# def setup():
#     # 시리얼 통신 초기화
#     ser.flushInput()
#     ser.flushOutput()
#     time.sleep(2)
#
#
# def loop():
#     setup()
#     print("Sending...")
#     send_cmd()
#     time.sleep(1)
#
#     # 데이터 수신
#     recv_cnt = 0
#     while recv_cnt < 8:
#         if ser.in_waiting > 0:
#             Receive_Buff[recv_cnt] = ord(ser.read(1))
#             recv_cnt += 1
#
#     if checksum_cal() == Receive_Buff[7]:
#         PPM_Value = (Receive_Buff[3] << 8) | Receive_Buff[4]
#         print("PPM:", PPM_Value)
#     else:
#         print("CHECKSUM Error")
#
#     time.sleep(1)
#
#
# if __name__ == "__main__":
#     Receive_Buff = [0] * 8
#     while True:
#         loop()