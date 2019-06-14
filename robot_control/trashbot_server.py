
from trashbot import TrashBot
from flask import Flask,request

bot = TrashBot()
app = Flask(__name__)

@app.route('/move', methods=['GET'])
def move():
    bot.move()
@app.route('/stop', methods=['GET'])
def stop():
    bot.stop()
@app.route('/turn_left', methods=['GET'])
def turn_left():
    bot.rotate_left()
@app.route('/turn_right', methods=['GET'])
def turn_right():
    bot.rotate_right()

@app.route('/throw_trash', methods=['GET'])
def throw_trash():
    bot.throw_trah()

@app.route('/throw_reset', methods=['GET'])
def throw_reset():
    bot.reset_throw()
@app.route('/say', methods=['GET'])
def say():
    bot.say()


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5005)