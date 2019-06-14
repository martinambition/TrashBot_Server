import requests
class BotFacade():
    def __init__(self,base_url):
        self.base_url =base_url
    def move_bot(self):
        print('move')
        requests.get(self.base_url + '/move',verify=False)

    def turn_left(self):
        print('left')
        requests.get(self.base_url + '/turn_left',verify=False)

    def turn_right(self):
        print('right')
        requests.get(self.base_url + '/turn_right', verify=False)

    def stop(self):
        requests.get(self.base_url + '/stop', verify=False)

