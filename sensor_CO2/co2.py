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
        if ppm <= 400:
            return 0  # 좋음
        elif ppm <= 1000:
            return 1  # 평범
        elif 1000 < ppm <= 2000:
            return 2  # 주의
        elif 2000 < ppm <= 5000:
            return 3  # 피로 유발
        else:
            return 4  # 위험

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


#from co2 import CO2Sensor
#
#sensor = CO2Sensor()
#sensor.run()
