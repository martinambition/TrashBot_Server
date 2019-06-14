from flask import Flask, request, jsonify
# import dev3.dev as 
from robot import Robot

app = Flask(__name__)
robot = Robot(mediumMotorOut='outA')

@app.route("/say", methods = ['GET'])
def say():
    msg = request.args.get("msg", default = 'hi')
    robot.say(msg)
    return 'done saying'

@app.route("/status", methods=['GET'])
def getStatus():
    return  jsonify({
        'position_sp': robot.getCurrentPosition()
    })

@app.route("/hold", methods=['GET'])
def hold():
    seconds = request.args.get("sec", default=1, type=float)
    angle = request.args.get("angle", default=90, type=float)
    robot.hold(seconds=seconds, angle=angle)
    return 'done holding'

# robot.say("I am ready")
# app.run()