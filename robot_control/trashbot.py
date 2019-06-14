#!/usr/bin/env python3

"""
Implementation of GRIPP3R
- use the remote control on channel 1 to drive the robot
- use the remote control on channel 4 to open/close the claw
    - press the top left button to close
    - press the bottom left button to open
- If GRIPP3R drives into something solid enough to press the
  TouchSensor underneath the claw will close. Once close the
  remote control must be used to open it.
"""

import logging
import signal

import sys
from ev3dev2.motor import OUTPUT_A, OUTPUT_B, OUTPUT_C, MediumMotor,SpeedPercent
from ev3dev2.control.rc_tank import RemoteControlledTank
from ev3dev2.sensor.lego import TouchSensor
from threading import Thread, Event
from time import sleep
from ev3dev2.motor import MoveTank
from ev3dev2.sound import Sound

log = logging.getLogger(__name__)


class MonitorTouchSensor(Thread):
    """
    A thread to monitor Gripper's TouchSensor and close the gripper when
    the TouchSensor is pressed
    """

    def __init__(self, parent):
        Thread.__init__(self)
        self.parent = parent
        self.shutdown_event = Event()
        self.monitor_ts = Event()

    def __str__(self):
        return "MonitorTouchSensor"

    def run(self):

        while True:

            if self.monitor_ts.is_set() and self.parent.ts.is_released:

                # We only wait for 1000ms so that we can wake up to see if
                # our shutdown_event has been set
                if self.parent.ts.wait_for_pressed(timeout_ms=1000):
                    self.parent.claw_close(True)

            if self.shutdown_event.is_set():
                log.info('%s: shutdown_event is set' % self)
                break


class TrashBot(MoveTank):
    """
    To enable the medium motor toggle the beacon button on the EV3 remote.
    """
    CLAW_DEGREES_OPEN = 225
    CLAW_DEGREES_CLOSE = 920
    CLAW_SPEED_PCT = 50

    def __init__(self, left_motor_port=OUTPUT_B, right_motor_port=OUTPUT_C, medium_motor_port=OUTPUT_A):
        MoveTank.__init__(self, left_motor_port, right_motor_port)
        self.set_polarity(MediumMotor.POLARITY_NORMAL)
        self.medium_motor = MediumMotor(medium_motor_port)
        self.sound = Sound()
        #self.ts = TouchSensor()
        #self.shutdown_event = Event()


    def signal_term_handler(self, signal, frame):
        log.info('Caught SIGTERM')
        self.shutdown_robot()

    def signal_int_handler(self, signal, frame):
        log.info('Caught SIGINT')
        self.shutdown_robot()

    def move(self):
        print('Move')
        self.on(SpeedPercent(80), SpeedPercent(80))
    def rotate_left(self):
        print('Left')
        self.on_for_rotations(SpeedPercent(50), SpeedPercent(70), 2)
    def rotate_right(self):
        print('Right')
        self.on_for_rotations(SpeedPercent(70), SpeedPercent(50), 2)
    def throw_trah(self):
        print('Throw')
        self.medium_motor.run_to_rel_pos(speed_sp=400, position_sp=75)

    def reset_throw(self):
        print('Reset')
        self.medium_motor.run_to_rel_pos(speed_sp=400, position_sp=-75)

    def say(self):
        self.sou==("speech.wav")