import tornado
import tornado.web
import tornado.websocket
import os
import cv2
import base64
from threading import Thread, Lock


class SocketServer():
    def __init__(self,port,q):
        self._port = port
        _ChatSocketHandler.queue =q

    def __call__(self):
        app = _Application()
        app.listen(self._port)
        tornado.ioloop.IOLoop.current().start()


class _Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", _ChatSocketHandler)
        ]
        settings = dict(
            xsrf_cookies=True,
        )
        super(_Application, self).__init__(handlers, **settings)

class _ChatSocketHandler(tornado.websocket.WebSocketHandler):
    queue = None
    waiters = set()
    cache = []
    cache_size = 200
    locker = Lock()

    def check_origin(self, origin):
        return True

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        _ChatSocketHandler.locker.acquire()
        _ChatSocketHandler.waiters.add(self)
        print("Connected");
        _ChatSocketHandler.locker.release()

    def on_close(self):
        _ChatSocketHandler.locker.acquire()
        _ChatSocketHandler.waiters.remove(self)
        _ChatSocketHandler.locker.release()

    @classmethod
    def update_cache(cls, chat):
        cls.cache.append(chat)
        if len(cls.cache) > cls.cache_size:
            cls.cache = cls.cache[-cls.cache_size:]
    @classmethod
    def send(cls, chat):
        _ChatSocketHandler.locker.acquire()
        for waiter in cls.waiters:
            try:
                waiter.write_message(chat)
            except:
                pass
        _ChatSocketHandler.locker.release()

    def on_message(self, message):
        if _ChatSocketHandler.queue:
            _ChatSocketHandler.queue.put(message)
