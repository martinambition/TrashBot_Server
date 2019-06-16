import requests
class BotFacade():
    def __init__(self,base_url):
        self.base_url =base_url
        self.enable = True
    def move_bot(self):
        print('move')
        if self.enable:
            requests.get(self.base_url + '/move',verify=False)

    def turn_left(self):
        print('left')
        if self.enable:
            requests.get(self.base_url + '/turn_left',verify=False)

    def turn_right(self):
        print('right')
        if self.enable:
            requests.get(self.base_url + '/turn_right', verify=False)

    def turn_around_left(self):
        print('left_center')
        if self.enable:
            requests.get(self.base_url + '/center_left', verify=False)

    def turn_around_right(self):
        print('right_center')
        if self.enable:
            requests.get(self.base_url + '/center_right', verify=False)

    def throw_trash(self):
        print('throw_trash')
        if self.enable:
            requests.get(self.base_url + '/throw_trash', verify=False)

    def throw_reset(self):
        print('throw_reset')
        if self.enable:
            requests.get(self.base_url + '/throw_reset', verify=False)

    def stop(self):
        if self.enable:
            requests.get(self.base_url + '/stop', verify=False)

