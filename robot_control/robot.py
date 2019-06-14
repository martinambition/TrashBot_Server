import ev3dev.ev3 as ev3
import time

class Robot:
    def __init__(self, mediumMotorOut = 'outA'):
        self.mediumMotorOut= mediumMotorOut
        self.mediumMotor = ev3.Motor(mediumMotorOut)
        self.touchSensor = ev3.TouchSensor()
        self.colorSensor = ev3.ColorSensor()

    def rotateMediumMotor(self, position_sp=0):
        self._configMotor()
        self.mediumMotor.run_to_abs_pos(position_sp=position_sp)

    def say(self, msg = "Hello"):
        ev3.Sound.speak(msg).wait()

    def hold(self, seconds=1, angle=90):
        self._configMotor()
        self.mediumMotor.run_to_rel_pos(position_sp=angle, stop_action="hold")
        # time.sleep(seconds)
        # self.mediumMotor.run_to_rel_pos(position_sp=-angle, stop_action='brake')

    def _configMotor(self, speed_sp=500):
        self.mediumMotor.speed_sp = speed_sp
    
    def getCurrentPosition(self):
        return self.mediumMotor.position_sp
    
    def shine(self):
        # while True:
            ev3.Leds.set_color(ev3.Leds.LEFT, (ev3.Leds.GREEN, ev3.Leds.RED)[self.touchSensor.value()])

    def detectColor(self):
        while True:
            colorMap = {
                ev3.ColorSensor.COLOR_BLACK: 'black',
                ev3.ColorSensor.COLOR_RED: 'red',
                ev3.ColorSensor.COLOR_YELLOW: 'yellow',
            }
            color = self.colorSensor.color
            if color in colorMap:
                msg = colorMap[color]
                print('color: ', msg)
                # self.say(msg=msg)
            # else:
            #     print('Unknown color: ', color)
            