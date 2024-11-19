from re import I, S, T
from kivy.app import ObjectProperty
from numpy.linalg import inv
from numpy.random import f
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivy.uix.scrollview import ScrollView
from kivy.config import Config
from kivy.uix.floatlayout import FloatLayout
from garden_matplotlib.backend_kivyagg import FigureCanvasKivyAgg
#from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivymd.uix.textfield import MDTextField
from kivy.uix.textinput import TextInput
from kivymd.uix.boxlayout import BoxLayout
from plyer import filechooser
from kivy.graphics.texture import Texture
from kivy.graphics import Line
from sort import Sort
from kivymd.uix.card import MDCard
from kivy.uix.image import Image
from kivy.core.window import Window
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp
from kivy.clock import mainthread

import sqlite3

import time
import math
import onnxruntime as rt
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

import yolov5
import arms
import theft

plt.ioff()

from face_detector import FaceDetector

#Window.borderless = True

Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '640')
Config.set('graphics', 'resizable', False)
Config.set('input', 'mouse', 'mouse,disable_multitouch')
Config.write()

import matplotlib as mpl

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR



mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

class Yolo5:
    def __init__(self) -> None:
        self.model = rt.InferenceSession(r"yolov5s.onnx")
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.names = yolov5.load_classes(r"coco.names")
        self.invasion = 0
        self.trespassing = 0
        self.unwanted_occurence = 0
        self.count_in = 0
        self.count_out = 0


    def yolov5_inference(self, 
                        cords, 
                        frame, 
                        invasion, 
                        occupancy, 
                        num_people,
                        trespassing,
                        unwanted_occurence):
        
        img_bgr = frame

        img_show = frame, 0, 0

        probability_outs = self.invasion, self.trespassing, self.unwanted_occurence

        #self.num_people = num_people

        img = cv2.resize(img_bgr, (self.input_h, self.input_w))
        img = img.astype('float32') / 255.
        img = img.transpose(2,0,1)
        img = img.reshape(*self.input_shape)

        detections = np.empty((0, 5))

        raw_result = self.model.run([], {self.input_name:img})
        
        prediction = raw_result[0]
 
        xc = prediction[..., 4] > 0.3

        min_wh , max_wh = 2, 7680

        x = prediction[0]

        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[0]]

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        boxes = x[:, :4]

        boxes = yolov5.xywh2xyxy(boxes)
        classes_score = x[:, 5:]

        det_result_oc = []

        det_result_in = []

        animal = ["bird","cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
       
        if occupancy:
            for cls in range(80):
                scores = classes_score[:, cls].flatten()
                pick = yolov5.nms(boxes, scores, 0.6, 0.4)
                for i in range(len(pick)):
                    det_result_oc.append([cls, scores[pick][i], boxes[pick][i]])
            img_show = yolov5.plot_boxes_cv2(img_bgr, det_result_oc, detections, self.names)

        for cls in range(80):
            scores = classes_score[:, cls].flatten()
            for i in range(len(scores)):
                det_result_in.append([cls, scores[i]])

        for det in det_result_in:
            if det[1] > 0.6:
                cls_id = det[0]
                label = self.names[cls_id]

                if label == "person":
                    if invasion:
                        num_people += 1
                        self.invasion = num_people / 20

                    if trespassing:
                        self.trespassing = det[1]

                if label in animal:
                    if unwanted_occurence:
                        self.unwanted_occurence = det[1]

            probability_outs = self.invasion, self.trespassing, self.unwanted_occurence 

        return img_show, probability_outs




class Arms:
    def __init__(self):
        self.model = rt.InferenceSession(r"arms.onnx")
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.names = arms.load_classes(r"arms.names")
        self.arms_det = 0
    
    def arms_inference(self, frame):
        img_bgr = frame

        img_show = frame, 0


        img = cv2.resize(img_bgr, (self.input_h, self.input_w))
        img = img.astype('float32') / 255.
        img = img.transpose(2,0,1)
        img = img.reshape(*self.input_shape)

        detections = np.empty((0, 5))

        raw_result = self.model.run([], {self.input_name:img})
        
        prediction = raw_result[0]

        xc = prediction[..., 4] > 0.3

        min_wh , max_wh = 2, 7680

        x = prediction[0]

        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[0]]

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        boxes = x[:, :4]

        boxes = arms.xywh2xyxy(boxes)
        classes_score = x[:, 5:]

        det_result = []

        for cls in range(1):
            scores = classes_score[:, cls].flatten()
            for i in range(len(scores)):
                det_result.append([cls, scores[i]])

        for det in det_result:
            if det[1] > 0.6:
                cls_id = det[0]
                label = self.names[cls_id]

                if label == "gun":
                    print("gun: ", det[1])
                    self.arms_det = det[1]
        
        return self.arms_det


class Theft:
    def __init__(self):
        self.model = rt.InferenceSession(r"yolov5s.onnx")
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.names = theft.load_classes(r"coco.names")
        self.theft_prob = 0


    def theft_inference(self, frame):
        
        img_bgr = frame


        img = cv2.resize(img_bgr, (self.input_h, self.input_w))
        img = img.astype('float32') / 255.
        img = img.transpose(2,0,1)
        img = img.reshape(*self.input_shape)

        detections = np.empty((0, 5))

        raw_result = self.model.run([], {self.input_name:img})
        
        prediction = raw_result[0]

        xc = prediction[..., 4] > 0.3

        min_wh , max_wh = 2, 7680

        x = prediction[0]

        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[0]]

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        boxes = x[:, :4]

        boxes = yolov5.xywh2xyxy(boxes)
        classes_score = x[:, 5:]

        det_result = []

        for cls in range(80):
            scores = classes_score[:, cls].flatten()
            pick = theft.nms(boxes, scores, 0.6, 0.4)
            for i in range(len(pick)):
                det_result.append([cls, scores[pick][i], boxes[pick][i]])

        img_show, self.theft_prob = theft.plot_boxes_cv2(img_bgr, det_result, detections, 0,self.names)

        return img_show, self.theft_prob



class GeneralSuspiciousBehaviour:
    def __init__(self, frame, ret) -> None:
        self.frame1 = frame
        self.ret = ret
        self.rect_count = 0

        self.prvs_gray = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)

    def detect_fast_moving(self, frame):
        if self.ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_diff = cv2.absdiff(self.prvs_gray, gray)

            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if cv2.contourArea(contour) > 500:                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    self.rect_count += 1
            
            if self.rect_count:
                self.rect_count = self.rect_count/10

            self.prvs_gray = gray

        return frame, self.rect_count
        
        
class FaceMaskDetection:
    def __init__(self) -> None:
        self.model = rt.InferenceSession("mask_detector.onnx", None)
        self.detection_model = FaceDetector("scrfd_500m.onnx")
        self.width = 128
        self.height = 128
        self.pred_prob = 0


    def get_optimal_font_scale(self, text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize =  cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10

        return 1


    def make_detections(self, frames):

        faces, inference_time, cropped_face = self.detection_model.inference(frames)

        try:
            bboxes = []
            for face in faces:
                face_img = face.cropped_face
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                face_img = cv2.resize(face_img, (self.width, self.height))
                face_img = face_img.astype(np.float32)
                face_img = face_img / 255.0
                face_img = face_img.reshape(1, self.width, self.height, 3)

                model_predict = self.model.run(['dense_1'], {'conv2d_input': face_img})
                max_index = np.argmax(model_predict)

                if max_index == 0:
                    self.pred_prob = np.max(model_predict)
                    text = "Mask"
                    color = (0, 255, 0)

                else:
                    text = "No Mask"
                    color = (0, 0, 255)

                font_size = self.get_optimal_font_scale(text, (int(face.bbox[3]) - int(face.bbox[1]))/3)
                cv2.rectangle(frames, 
                             (int(face.bbox[0]), 
                              int(face.bbox[1])), 
                             (int(face.bbox[2]), 
                              int(face.bbox[3])), color, 2)
                cv2.putText(frames,
                            text, 
                            (int(face.bbox[0]), 
                            int(face.bbox[1])-6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_size, 
                            color, 2, cv2.LINE_AA)

        except:
            text = 'Face not Detected'
            font_size = self.get_optimal_font_scale(text, frames.shape[1] // 6)
            #cv2.putText(frames, text, (int(face.bbox[0]), int(face.bbox[1])-6), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2, cv2.LINE_AA)


        return frames, self.pred_prob


class MyTable(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__(**kwargs)
        self.mainwid = mainwid


        self.row_data = [("None", "None")]

        self.data_tables = MDDataTable(
            size_hint=(0.32, 0.5),
            pos = (680, 41),
            use_pagination=False,
            rows_num= 500,
            check=False,
            elevation=2,
            # name column, width column, sorting function column(optional), custom tooltip
            
            column_data=[
                ("Detections", dp(30)),
                ("time", dp(30))
            ],

            row_data=self.row_data,

        )

        self.data_tables.bind(on_row_press=self.on_row_press)

        self.add_widget(self.data_tables)


    def on_row_press(self, instance_table, instance_row):

        row_num = int(instance_row.index/len(instance_table.column_data))
        row_data = instance_table.row_data[row_num]
        _, row_texts = row_data
        
        try:
            self.mainwid.player_wid.set_frame_from_table(row_texts)

        except:
            pass


    def add_row(self, detection_incident, frame_now):
        self.data_tables.add_row((detection_incident, str(frame_now)))


class ActivityChart(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__(**kwargs)
        self.mainwid = mainwid

        labels = ["Animals", "Trespassing", "FaceMask", "Invasion", "Theft", "Arms", "FastMoving"]

        self.data = [5, 6, 6, 7, 9, 3, 1]
        self.fig, self.ax = plt.subplots(subplot_kw=dict(polar=True))
        self.ax.yaxis.label.set_color('red')
        self.ax.xaxis.label.set_color('red')
        self.fig.patch.set_facecolor((0.0705882352941176, 0.0705882352941176, 0.0705882352941176, 1.0))
        self.angles = np.linspace(0, 2*np.pi, len(self.data), endpoint=False)
        self.plot, = self.ax.plot(self.angles, self.data, linewidth=2, color="white")
        self.fill = None
        self.ax.set_thetagrids(self.angles * 180 / np.pi, labels)
        self.ax.set_facecolor("grey")


        canvas = FigureCanvasKivyAgg(figure=self.fig)
        box = self.ids.box
        box.add_widget(canvas)
        Clock.schedule_interval(self.update_animation, 2)

    def update_animation(self, dt):
        #self.data = np.random.randint(2, 7, size=len(self.angles))
        self.data = [0,0,0,0,0,0,0]
        self.data[0] = self.mainwid.player_wid.unwanted_occurence_prob * 10
        self.data[1] = self.mainwid.player_wid.trespassing_prob * 10
        self.data[2] = self.mainwid.player_wid.pred_prob * 10
        self.data[3] = self.mainwid.player_wid.invasion_prob * 10
        self.data[4] = self.mainwid.player_wid.theft_prob
        self.data[5] = self.mainwid.player_wid.arms_prob * 10
        self.data[6] = self.mainwid.player_wid.fast_prob * 10
        self.values = np.concatenate((self.data, [self.data[0]]))
        self.plot.set_xdata(np.concatenate((self.angles, [self.angles[0]])))
        if self.fill is not None:
            for f in self.fill:
                f.remove()
       
        self.plot.set_ydata(self.values)
        self.fill = self.ax.fill(self.angles, self.data, color="red", alpha=0.25)
        self.fig.canvas.draw_idle()


class LoadFile(FloatLayout):
    def __init__(self, mainwid, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mainwid = mainwid

    def select_file(self):
        self.reset_prams()
        try:
            filechooser.open_file(on_selection=self.selected, multiple=False, filters=["*.mp4", "*.mkv", "*mpeg"])
        except:
            pass

    def selected(self, selection):
        self.file_path = '\\'.join(selection)
        file_path_f = os.path.basename(self.file_path)
        self.mainwid.player_wid.setup_file(self.file_path)
        file_label = self.ids.myfile
        file_label.text = file_path_f
        print(file_path_f)
        return self.file_path


    def play(self):
        self.mainwid.player_wid.play()

    def forward(self):
        self.mainwid.player_wid.forward()

    def set_detection_status(self):
        self.mainwid.player_wid.set_detection_status()

    def reset_prams(self):
        self.mainwid.player_wid.reset_params()


class VideoShow:
    def __init__(self, frame=None, image_widget=None):
        self.frame = frame
        self.stopped = False
        self.image_widget = image_widget

    def start(self):
        Clock.schedule_interval(self.start_s, 0)
        return self

    def start_s(self, *args):
        Thread(target=self.show, args=()).start()

    @mainthread
    def show(self, *args):
        self.image_show(self.frame, self.image_widget)

    def stop(self):
        self.stopped = True

    def image_show(self, frame, image_widget):
        buf = cv2.flip(frame, 0)
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        video = image_widget
        video.texture = image_texture

class VideoGet:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.ret = self.grabbed


    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()

            else:
                self.grabbed, self.frame = self.stream.read()

    def stop(self):
        self.stopped = True


class PlayerWid(FloatLayout):
    image_texture = ObjectProperty(None)
    image_capture = ObjectProperty(None)
    timeSlider = ObjectProperty(None)

    frame_time_count_s =  ObjectProperty(None)

    linePlay = False

    def __init__(self, mainwid, **kwargs):
        super().__init__(**kwargs)
        self.mainwid = mainwid
        self.flagPlay = False
        self.now_frame = 0
        self.frame_time_value = 0
        self.image_index = []
        self.image_generator = None
        self.frame_out = None

        self.occupancy = False
        self.suspicious_behaviour = False
        self.facemask = False
        self.invasion = False
        self.trespassing = False
        self.unwanted_occurence = False
        self.arms = False
        self.theft = False

        self.invasion_prob = 0
        self.trespassing_prob = 0
        self.unwanted_occurence_prob = 0
        self.pred_prob = 0
        self.theft_prob = 0
        self.fast_prob = 0
        self.arms_prob = 0

        self.count_in = 0
        self.count_out = 0
        self.my_card = self.ids['my_card']
        self.my_coords = []
        self.line_coordinates = []
        self.linePlay = False
        self.set_image_layout = False
        self.slider_updated = False
        self.slider_updated_frame = 0

        self.my_image = Image()
        self.image_layout = self.ids['image_layout']
        self.image_widget = self.my_image

        self.frame_time = 0
        self.rt = 0

        self.thread_on = False

        self.current_time = "00:00"

        self.detection_incident = "None"

        self.count_in_label = self.ids['in']
        self.count_out_label = self.ids['out']

        self.start_time_invasion = time.time()
        self.start_time_tresspassing = time.time()
        self.start_time_theft = time.time()
        self.start_time_arms = time.time()
        self.start_time_fast = time.time()
        self.start_time_unwanted_occurrence = time.time()
        self.start_time_facemask = time.time()


    def setup_file(self, filename):
        self.image_capture = cv2.VideoCapture(filename)
        self.sliderSetting()
        self.filename = filename

    def sliderSetting(self):
        print("reading frames")
        self.count_frames = self.image_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_time = self.image_capture.get(cv2.CAP_PROP_FPS)
        self.ids["timeSlider"].max = self.count_frames/self.frame_time

        time_duration = self.count_frames/self.frame_time
        time_duration_minutes = time_duration//60
        if time_duration >=60:
            time_duration_seconds = time_duration%60

        else:
            time_duration_seconds = time_duration

        self.time_duration_s = f"{str(round(time_duration_minutes))}:{str(round(time_duration_seconds)).zfill(2)}"
        self.ids["my_labels"].text = f"{self.current_time}/{self.time_duration_s}"

    def forward(self):
        self.now_frame += 40
        if self.now_frame >= self.count_frames:
            self.now_frame = 0

        self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, self.now_frame)


    def rewind(self):
        self.now_frame -= 40
        if self.now_frame <=40:
            self.now_frame = 0

        self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, self.now_frame)

    def reload_play(self):
        self.ids["timeSlider"].value = 0
        self.now_frame = 0
        self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, self.now_frame)
        self.mainwid.activity_chart.row_data = [("None", "None")]

    def play(self):
        self.flagPlay = not self.flagPlay
        self.my_icon = self.ids["plays"]
        if self.my_icon.icon != 'replay':
            if self.flagPlay == True:
                self.my_icon.icon = 'pause'

                # if self.slider_updated:
                #     self.now_frame = self.slider_updated_frame

                self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, self.now_frame)
                #self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                my_fps = self.image_capture.get(cv2.CAP_PROP_FPS)
                print(my_fps)

                if self.thread_on == False:
                    Clock.schedule_interval(self.update, 1.0 / self.image_capture.get(cv2.CAP_PROP_FPS))
                else:
                    self.update(self.rt)
            else:
                self.my_icon.icon = 'play-outline'
                Clock.unschedule(self.update)

        else:
            self.my_icon.icon = 'play-outline'
            self.reload_play()
    
    def update(self, dt):    
        if self.flagPlay:                        

            if self.thread_on == False:
            
                ret, frame = self.image_capture.read()
                #ret, frame = self.mainwid.video_get.ret, self.mainwid.video_get.frame
                if ret:
                    frame = self.update_image(frame)
                    time = self.image_capture.get(cv2.CAP_PROP_POS_FRAMES)
                    self.ids["timeSlider"].value = time/self.frame_time
                    time_value = time/self.frame_time
                    time_value_minutes = time_value//60
                    if time_value >=60:
                        time_value_seconds = time_value%60

                    else:
                        time_value_seconds = time_value


                    self.current_time = f"{str(round(time_value_minutes))}:{str(round(time_value_seconds)).zfill(2)}"
                    self.ids["my_labels"].text = f"{self.current_time}/{self.time_duration_s}"
                    self.now_frame = int(time)
                    self.time = time
                else:
                    self.my_icon.icon = 'replay'
                    Clock.unschedule(self.update)


            else:
                self.video_getter = VideoGet(self.filename).start()
                self.video_shower = VideoShow(self.video_getter.frame, self.image_widget).start()

                def ty():

                    while True:
                        if self.video_getter.stopped or self.video_shower.stopped:
                            self.video_shower.stop()
                            self.my_icon.icon = 'replay'
                            self.video_getter.stop()
                            break

                        frame = self.video_getter.frame
                        frame = self.update_image(frame)
                        self.video_shower.frame = frame
                        #self.video_shower.image_widget = self.image_widget

                        time = self.video_getter.stream.get(cv2.CAP_PROP_POS_FRAMES)
                        #self.ids["timeSlider"].value = time/self.frame_time

                        time_value = time/self.frame_time

                        self.ids["timeSlider"].value = time_value
                        time_value_minutes = time_value//60
                        if time_value >=60:
                            time_value_seconds = time_value%60

                        else:
                            time_value_seconds = time_value


                        self.current_time = f"{str(round(time_value_minutes))}:{str(round(time_value_seconds)).zfill(2)}"
                        self.ids["my_labels"].text = f"{self.current_time}/{self.time_duration_s}"
                        self.now_frame = int(time)
                        self.time = time

                Thread(target=ty).start()



    def sliderTouchMove(self, touch):

        if self.ids["timeSlider"].collide_point(*touch.pos):
            self.flagPlay = False

            Clock.schedule_interval(self.sliderUpdate, 0)
            
            if self.my_icon.icon == 'replay':
                self.my_icon.icon = 'play-outline'


        else:
            self.flagPlay = True

    def sliderUpdate(self, dt):
        if not self.flagPlay:
            self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.ids["timeSlider"].value * self.frame_time))
            ret, frame = self.image_capture.read()
    #            ret, frame = self.mainwid.video_get.ret, self.mainwid.video_get.frame
            if ret:
                #self.update_image(frame)
                self.now_frame = int(self.ids["timeSlider"].value * self.frame_time)


    def suspicious_behaviour_p(self, frame_org, *args):
        ret, frame2 = self.image_capture.read()
        if ret:
            frame2 = cv2.resize(frame2, (540, 320))

            frs = GeneralSuspiciousBehaviour(frame_org, ret)

            self.frame_out = frs.detect_fast_moving(frame2)

        return self

    def slider_play(self, touch):
        if self.ids["timeSlider"].collide_point(*touch.pos):
            self.slider_play_update(0)
            Clock.schedule_interval(self.slider_play_update, 0)

    def slider_play_update(self, dt):
        if self.flagPlay == False:
            self.flagPlay = True


    def facemask_p(self, frame_org):
        self.frame_out, pred_prob = self.mainwid.facemask_detector.make_detections(frame_org)
        self.pred_prob = pred_prob


    def update_image(self, frame):
        frame_change = False
        
        if self.set_image_layout == False:
            self.image_layout.add_widget(self.my_image)
            self.set_image_layout = True

        frame_org = cv2.resize(frame, (540, 320))

        frame_out = []

        if self.suspicious_behaviour:
            ret, frame2 = self.image_capture.read()
            if ret:
                frame2 = cv2.resize(frame2, (540, 320))

                frs = GeneralSuspiciousBehaviour(frame_org, ret)

                frame, fast_prob = frs.detect_fast_moving(frame2)
                if fast_prob > 1:
                    self.fast_prob = 1
                else:
                    self.fast_prob = fast_prob


            frame_change = True

        if self.facemask:
            face_mask_det, pred_prob = self.mainwid.facemask_detector.make_detections(frame_org)
            self.pred_prob = pred_prob
            frame_change = True

        if self.arms:
            arms_prob = self.mainwid.arms.arms_inference(frame_org)
            self.arms_prob = arms_prob
            frame_change = True

        if self.occupancy or self.invasion or self.trespassing or self.unwanted_occurence:
            num_people = 0
            img_show, probability_outs = self.mainwid.yolov5.yolov5_inference(self.my_coords,
                                                                frame_org, 
                                                                self.invasion, 
                                                                self.occupancy, 
                                                                num_people,
                                                                self.trespassing,
                                                                self.unwanted_occurence)

            frame, count_in, count_out = img_show
            self.count_in_label.text = str(count_in)
            self.count_out_label.text = str(count_out)
            self.invasion_prob, self.trespassing_prob, self.unwanted_occurence_prob =  probability_outs
            frame_change = True

        if self.theft:
            _, theft_prob = self.mainwid.theft.theft_inference(frame_org)

            self.theft_prob = theft_prob
            

        if frame_change == False:
            frame = frame_org


        if self.thread_on == False:
            buf = cv2.flip(frame, 0)
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            video = self.image_widget
            video.texture = image_texture


        if self.invasion_prob > 0.89:
            current_time_invasion = time.time()
            if current_time_invasion - self.start_time_invasion > 10:
                self.detection_incident = "invasion"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)

                self.start_time_invasion = current_time_invasion


        if self.trespassing_prob > 0.65:
            current_time_tres = time.time()
            if current_time_tres - self.start_time_tresspassing > 10:
                self.detection_incident = "Trespassing"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)
                
                self.start_time_tresspassing = current_time_tres

                

        if self.unwanted_occurence_prob > 0.65:
            current_time_unwanted_occurence = time.time()
            if current_time_unwanted_occurence - self.start_time_unwanted_occurrence > 10:
                self.detection_incident = "Animals"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)

                self.start_time_unwanted_occurrence = current_time_unwanted_occurence

        if self.arms_prob > 0.65:
            current_time_arms = time.time()
            if current_time_arms - self.start_time_arms > 10:
                self.detection_incident = "Arms"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)

                self.start_time_arms = current_time_arms

        if self.theft_prob > 0.1:
            current_time_theft = time.time()
            if current_time_theft - self.start_time_theft > 10:
                self.detection_incident = "Theft"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)
                
                self.start_time_theft = current_time_theft

        if self.fast_prob > 0.1:
            current_time_fast = time.time()
            if current_time_fast - self.start_time_fast > 10:
                self.detection_incident = "FastMoving"
                self.mainwid.mydata_table.add_row(self.detection_incident, self.current_time)
                
                self.start_time_fast = current_time_fast

        return frame

#        buf = cv2.flip(frame, 0)
#        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
#        image_texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
#        video = self.image_widget
#        video.texture = image_texture 

    def set_occupancy(self, checkbox, value):
        if value:
            self.occupancy = True
        else:
            self.occupancy = False

    def set_suspicious_behaviour(self, checkbox, value):
        if value:
            self.suspicious_behaviour = True
        else:
            self.suspicious_behaviour = False

    def set_facemask(self, checkbox, value):
        if value:
            self.facemask = True
        else:
            self.facemask = False

    def set_invasion(self, checkbox, value):
        if value:
            self.invasion = True
        else:
            self.invasion = False

    def set_trespassing(self, checkbox, value):
        if value:
            self.trespassing = True
        else:
            self.trespassing = False
        

    def set_unwanted_occurence(self, checkbox, value):
        if value:
            self.unwanted_occurence = True
        else:
            self.unwanted_occurence = False

    def set_arms(self, checkbox, value):
        if value:
            self.arms = True
        else:
            self.arms = False

    def set_theft(self, checkbox, value):
        if value:
            self.theft = True
        else:
            self.theft = False


    def set_frame_from_table(self, time):

        minutes, seconds = map(int, time.split(':'))
        time_seconds = minutes * 60 + seconds
        my_frames = time_seconds * self.frame_time
        self.now_frame = int(my_frames)

        print(self.now_frame)
        self.image_capture.set(cv2.CAP_PROP_POS_FRAMES, self.now_frame)

        Clock.schedule_once(self.update)
        
        #self.my_icon.icon = 'play-outline'


    def set_thread_on(self, switch, value):
        if value:
            if self.thread_on == False:
                self.thread_on = True
        else:
            self.thread_on = False


    def reset_params(self):
        #self.set_occupancy(_, False)
        self.ids["timeSlider"].value = 0
        self.theft = False
        self.unwanted_occurence = False
        self.trespassing = False
        self.invasion = False
        self.facemask = False
        self.occupancy = False
        self.suspicious_behaviour = False
        self.now_frame = 0


class MainWid(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolov5 = Yolo5()
        self.facemask_detector = FaceMaskDetection()
        self.arms = Arms()
        self.theft = Theft()
        self.activity_chart = ActivityChart(self)
        self.mydata_table = MyTable(self)
        self.load_file = LoadFile(self)
        self.add_widget(self.mydata_table)
        self.player_wid = PlayerWid(self)
#        self.video_get = VideoGet(self)
        self.add_widget(self.player_wid)
        self.add_widget(self.activity_chart)
        self.add_widget(self.load_file)
        


# class MainApp(MDApp):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.theme_cls.theme_style = "Dark"

#     def build(self):
#         return MainWid()


# if __name__ == '__main__':
#     MainApp().run()
