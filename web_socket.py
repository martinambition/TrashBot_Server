import tornado
import tornado.web
import tornado.websocket
import os
import cv2
import base64
from threading import Thread, Lock


class SocketServer():
    def __init__(self,port,q,outq,textqueue):
        self._port = port
        _ChatSocketHandler.inqueue =q
        _ChatSocketHandler.outqueue = outq
        _ChatSocketHandler.textqueue = textqueue

    def __call__(self):
        app = _Application()
        app.listen(self._port)
        tornado.ioloop.IOLoop.current().start()

class _Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", _ChatSocketHandler,dict(source='mobile')),
            (r"/web", _ChatSocketHandler,dict(source='web'))
        ]
        settings = dict(
            xsrf_cookies=True,
        )
        super(_Application, self).__init__(handlers, **settings)

class _ChatSocketHandler(tornado.websocket.WebSocketHandler):

    inqueue = None
    outqueue = None
    textqueue = None
    waiters_mobile = set()
    waiters_web = set()
    cache = []
    cache_size = 200
    locker = Lock()

    def initialize(self,source):
        self.source = source
    def check_origin(self, origin):
        return True

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        _ChatSocketHandler.locker.acquire()
        if self.source == "mobile":
            self.waiters_mobile.add(self)
            print("Mobile Connected");
        else:
            self.waiters_web.add(self)
            print("Web Connected");
        _ChatSocketHandler.locker.release()

    def on_close(self):
        _ChatSocketHandler.locker.acquire()
        if self.source == "mobile":
            _ChatSocketHandler.waiters_mobile.remove(self)
        else:
            _ChatSocketHandler.waiters_web.remove(self)
        _ChatSocketHandler.locker.release()

    @classmethod
    def update_cache(cls, chat):
        cls.cache.append(chat)
        if len(cls.cache) > cls.cache_size:
            cls.cache = cls.cache[-cls.cache_size:]
    @classmethod
    def send_frame(cls, frame):
        _ChatSocketHandler.locker.acquire()
        cnt = cv2.imencode('.jpg', frame)[1]
        str = base64.b64encode(cnt.ravel())
        for waiter in cls.waiters_web:
            try:
                waiter.write_message("data:image/jpg;base64," + str.decode('ascii'))
            except:
                pass
        _ChatSocketHandler.locker.release()

    @classmethod
    def send_text(cls, text):
        _ChatSocketHandler.locker.acquire()
        for waiter in cls.waiters_mobile:
            try:
                waiter.write_message(text)
            except:
                pass
        _ChatSocketHandler.locker.release()
    def on_message(self, message):
        if _ChatSocketHandler.inqueue:
            _ChatSocketHandler.inqueue.put(message)
            try:
                toweb = _ChatSocketHandler.outqueue.get(False)
                if toweb is not None:
                    _ChatSocketHandler.send_frame(toweb)
            except:
                pass

            try:
                text_to_mobile = _ChatSocketHandler.textqueue.get(False)
                if text_to_mobile is not None:
                    _ChatSocketHandler.send_text(text_to_mobile)
            except:
                pass