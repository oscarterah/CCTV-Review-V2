import argparse
import math
import numpy as np
import onnxruntime
import cv2
from sort import Sort
import threading

tracker = Sort(max_age=20)
totalCountUp = []
lock = threading.Lock()

def object_detection(frame, session, input_shape):
    img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    img = img.astype('float32') / 255.
    img = img.transpose(2, 0, 1)
    img = img.reshape(*input_shape)

    detections = np.empty((0, 5))
    raw_result = session.run([], {'input': img})

    prediction = raw_result[0]
    xc = prediction[..., 4] > 0.3

    min_wh, max_wh = 2, 7680
    x = prediction[0]
    x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
    x = x[xc[0]]

    x[:, 5:] *= x[:, 4:5]
    boxes = x[:, :4]
    boxes = xywh2xyxy(boxes)
    classes_score = x[:, 5:]

    det_result = []
    for cls in range(80):
        scores = classes_score[:, cls].flatten()
        pick = nms(boxes, scores, 0.6, 0.4)
        for i in range(len(pick)):
            det_result.append([cls, scores[pick][i], boxes[pick][i]])

    with lock:
        img_show, t_count = plot_boxes_cv2(frame, det_result, detections, names)
        cv2.imshow("Object Detection", img_show)
        totalCountUp.extend(t_count)

def video_capture_thread():
    video_capture = cv2.VideoCapture(video_input)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        object_detection(frame, session, input_shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov5s.onnx", help='model file')
    parser.add_argument('-i', '--input', type=str, default="0", help='input video or camera index')
    parser.add_argument('--names', type=str, default='coco.names', help='*.names path')
    args = parser.parse_args()
    model_file = args.model
    video_input = args.input

    names = load_classes(args.names)

    session = onnxruntime.InferenceSession(model_file, None)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    detection_thread = threading.Thread(target=video_capture_thread)
    detection_thread.start()
    detection_thread.join()

    cv2.destroyAllWindows()

