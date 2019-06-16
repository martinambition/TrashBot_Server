import cv2
import sys
import os
from multiprocessing import Process, Queue
import threading
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from web_socket import SocketServer
from bot_facade import BotFacade
from color_track import  ColorTrack

states= ['gopeople','trash_detecton','findbox', 'gobox' ,'return']
control_state = 'M'
current_state = ''
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph,sess):
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes',
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

text_queue = Queue()
def run():
    global current_state
    PATH_TO_FROZEN_GRAPH = '/Users/i303138/Documents/Learning/MachineLearning/Projects/Lego/tf_objdetect/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print("main process:",os.getpid())
    auto_model = False
    q = Queue()
    out_queue = Queue()
    global text_queue

    # start face recognization in child process
    socket_server =  SocketServer(50010,q,out_queue,text_queue)
    fw = Process(target=socket_server)
    fw.start()
    with detection_graph.as_default():
        with tf.Session() as sess:
            while (True):
                image_data = q.get(True)
                #Clear the queue
                while not q.empty():
                    image_data = q.get()
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
                #frame = rotateImage(frame,180)
                rec_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(rec_img, axis=0)

                output_dict = run_inference_for_single_image(image_np_expanded, detection_graph, sess)

                # height = frame.shape[0]
                # ratio = 500 / height
                # frame = cv2.resize(frame, (int(ratio * frame.shape[1]), int(ratio * frame.shape[0])))


                #
                #rec_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # path = "/Users/i303138/Documents/Learning/MachineLearning/Projects/Lego/Dataset/JPEGImages/Cola/IMG_20190607_105703.jpg"


                #height = frame.shape[0]
                # ratio = 500 / height
                #
                # frame = cv2.resize(frame, (int(ratio * frame.shape[1]), int(ratio * frame.shape[0])))
                boxs = output_dict['detection_boxes']
                scale_rect= np.array([0,0,0,0])
                detect_class = 0
                if len(boxs) > 0:
                    scores = output_dict['detection_scores'].tolist()
                    if len(scores) > 0 and scores[0] > 0.5:
                        rect = boxs[0]
                        im_width = frame.shape[1]
                        im_height = frame.shape[0]
                        # ymin,xmin,ymax,xmax,
                        scale_rect = np.array(
                            [rect[1] * im_width, rect[0] * im_height, rect[3] * im_width, rect[2] * im_height]).astype(int)
                        # scale_rect = (rect ).astype(int)
                        detect_class = output_dict['detection_classes'][0]
                        cv2.rectangle(frame, (scale_rect[0], scale_rect[1]), (scale_rect[2], scale_rect[3]), (255, 0, 0), 2)

                if auto_model:
                    bot_control(frame.shape, scale_rect, detect_class, frame)

                cv2.putText(frame, ('a' if auto_model else 'm'), (0, 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
                out_queue.put(frame)
                cv2.imshow('Video', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m') or key == ord('M'):
                    auto_model = not auto_model
                    if auto_model:
                        text_queue.put("iron")
                    current_state = 'gopeople'# 'gopeople'
                if not auto_model:
                    if key == ord('w') or key == ord('W'):
                        bot.move_bot()
                    if key == ord('a') or key == ord('A'):
                        bot.turn_around_left()
                    if key == ord('d') or key == ord('D'):
                        bot.turn_around_right()
                    if key == ord('r') or key == ord('R'):
                        text_queue.put("finished")
                        bot.throw_trash()
                        bot.throw_reset()

                    if key == ord('t') or key == ord('T'):
                        bot.throw_reset()

                    if key == ord('1'):
                        text_queue.put("iron")
                    if key == ord('2'):
                        text_queue.put("trash")
                    if key == ord('3'):
                        text_queue.put("finished")

bot = BotFacade("http://169.254.28.26:5005")
framecount = 0
colortrack = ColorTrack()
wait = 0
def bot_control(frame_shape,scale_rect,detect_class,frame):
    global framecount
    global bot
    global current_state
    global wait
    global colortrack
    global text_queue
    framecount = framecount + 1

    if (framecount %5 == 0):# Try to cal per 5 frames.
        framecount = 0
        if current_state == 'gopeople':
            threash_hold = 50;
            center = frame_shape[1]/2
            left_limit = center - threash_hold
            right_limit = center + threash_hold

            if detect_class == 1:
                person_center = (scale_rect[2] - scale_rect[0])/2 + scale_rect[0]
                if person_center> right_limit:
                    bot.turn_right()
                elif person_center < left_limit:
                    bot.turn_left()
                else:
                    bot.move_bot()
        if current_state == 'gopeople' and detect_class > 1 :
            text_queue.put("trash")
            current_state = 'trash_detecton'
        if current_state == 'trash_detecton':
            wait = wait +1
            if wait == 2:
                wait = 0
                current_state = 'findbox'
        if current_state == 'findbox':#Looking for box
            x, y, w, h, are = colortrack.detect_red(frame)
            if are < 10:
                bot.turn_around_right()
            if are >= 10:
                current_state = "gobox"
        if current_state == "gobox":
            x, y, w, h, are = colortrack.detect_red(frame)
            if are/(frame_shape[1]*frame_shape[0]) >0.11:
                bot.throw_trash()
                bot.throw_reset()
                text_queue.put("finished")
                current_state = "end"
            elif are >=10:
                threash_hold = 50;
                center = frame_shape[1] / 2
                left_limit = center - threash_hold
                right_limit = center + threash_hold
                box_center = x+w/2
                if box_center > right_limit:
                    #print('right')
                    bot.turn_right()
                elif box_center < left_limit:
                    #print('left')
                    bot.turn_left()
                else:
                    #print('move')
                    bot.move_bot()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))
    run()