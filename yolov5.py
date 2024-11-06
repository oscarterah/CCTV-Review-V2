import argparse
import math
import numpy as np
import onnxruntime
import cv2
from sort import Sort

tracker = Sort(max_age=20)


totalCountUp = []
totalCountDown = []

CrossedLine1 = None

count_out = 0
count_in = 0

passed_line1 = False
passed_line2 = False

def osc():
	print('in yolov5')

def sort_score_index(scores, threshold=0.0, top_k=0, descending=True):
	score_index = []
	for i, score in enumerate(scores):
		if (threshold > 0) and (score > threshold):
			score_index.append([score, i])
	if not score_index:
		return []

	np_scores = np.array(score_index)
	if descending:
		np_scores = np_scores[np_scores[:, 0].argsort()[::-1]]
	else:
		np_scores = np_scores[np_scores[:, 0].argsort()]

	if top_k > 0:
		np_scores = np_scores[0:top_k]
	return np_scores.tolist()


def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3, top_k=0):
	if scores is not None:
		scores = sort_score_index(scores, score_threshold, top_k)
		if scores:
			order = np.array(scores, np.int32)[:, 1]
		else:
			return []
	else:
		y2 = boxes[:3]
		order = np.argsort(y2)

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	areas = (x2 - x1) * (y2 - y1)
	keep = []

	while len(order) > 0:
		idx_self = order[0]
		idx_other = order[1:]
		keep.append(idx_self)

		xx1 = np.maximum(x1[idx_self], x1[idx_other])
		yy1 = np.maximum(y1[idx_self], y1[idx_other])
		xx2 = np.minimum(x2[idx_self], x2[idx_other])
		yy2 = np.minimum(y2[idx_self], y2[idx_other])
		w = np.maximum(0.0, xx2 - xx1)
		h = np.maximum(0.0, yy2 - yy1)

		inter = w * h
		over = inter / (areas[order[0]] + areas[order[1:]] - inter)

		inds = np.where(over <= iou_threshold)[0]
		order = order[inds + 1]

	return keep


def xywh2xyxy(x):
	y = np.zeros_like(x)
	y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
	y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
	y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
	y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
	return y


def load_classes(path):
	with open(path, 'r') as f:
		names = f.read().split('\n')
	return list(filter(None, names))


def plot_boxes_cv2(img, det_result, detections, class_names=None):
	global count_in
	global count_out
	global CrossedLine1
	global passed_line1
	global passed_line2
	img = np.copy(img)
	detections = detections
	colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

	def get_color(c, x, max_val):
		ratio = float(x) / max_val * 5
		i = int(math.floor(ratio))
		j = int(math.ceil(ratio))
		ratio = ratio - i
		r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
		return int(r * 255)

	w = img.shape[1] / 320.
	h = img.shape[0] / 320.
	

	limitsUp = [150, 130, 280, 130]

	limitsDown = [150, 140, 300, 140]


	label = ""

	tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

	for det in det_result:
		c1 = (int(det[2][0] * w), int(det[2][1] * h))
		c2 = (int(det[2][2] * w), int(det[2][3] * h))



		if det[1] > 0.5:  # Set your confidence threshold here
			rgb = (255, 0, 0) #blue color for box

			if class_names: 
				cls_conf = det[1]
				cls_id = det[0]
				label = class_names[cls_id]
	 #			 print("{}: {}".format(label, cls_conf))
				classes = len(class_names)
				offset = cls_id * 123457 % classes
				red = get_color(2, offset, classes)
				green = get_color(1, offset, classes)
				blue = get_color(0, offset, classes)
				rgb = (red, green, blue)

				tf = max(tl - 1, 1)
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				cc2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
				cv2.rectangle(img, c1, cc2, rgb, -1)
				cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

			if label == "person":
				x1, y1 = c1
				x2, y2 = c2

				img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), tl)
				conf = det[1]

				currentArray = np.array([x1, y1, x2, y2, conf])

				detections = np.vstack((detections, currentArray))

	
	resultsTracker = tracker.update(detections)

	img = cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
	img = cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
	crossed_lines = {}

	for result in resultsTracker:
		x1, y1, x2, y2, obj_id = result
		w1, h1 = x2 - x1, y2 - y1
			
		cx, cy = x1 + w1//2, y1 + h1//2

		cx = int(cx)
		cy = int(cy)

			  
		img = cv2.circle(img, (cx, cy), 5, (255, 0, 225), cv2.FILLED)

		if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 5 < cy < limitsUp[1] + 5:
			CrossedLine1 = True
			passed_line1 = True

		if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 5 < cy < limitsDown[1] + 5:
			CrossedLine1 = False
			passed_line2 = True
		if passed_line2 == True and passed_line1 == True:
			if CrossedLine1 == True and totalCountUp.count(id) ==  0:
				totalCountDown.append(obj_id)

			elif CrossedLine1 == False and totalCountDown.count(id) == 0:
				totalCountUp.append(obj_id)

			passed_line1 = False
			passed_line2 = False


	count_in = len(totalCountUp)
	count_out = len(totalCountDown)


	return img, count_in, count_out


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
	input_h = input_shape[2]
	input_w = input_shape[3]

	count_in = 0
	count_out = 0

	video_capture = cv2.VideoCapture(video_input)

	while True:
		ret, frame = video_capture.read()

		if not ret:
			break

		img_bgr = frame

		img = cv2.resize(img_bgr, (input_h, input_w))
		img = img.astype('float32') / 255.
		img = img.transpose(2, 0, 1)
		img = img.reshape(*input_shape)

		detections = np.empty((0, 5))
		raw_result = session.run([], {input_name: img})

		prediction = raw_result[0]
		xc = prediction[..., 4] > 0.3

		min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
		x = prediction[0]
		x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0	# width-height
		x = x[xc[0]]

		# Compute conf
		x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

		boxes = x[:, :4]
		boxes = xywh2xyxy(boxes)
		classes_score = x[:, 5:]

		det_result = []
		for cls in range(80):
			scores = classes_score[:, cls].flatten()
			pick = nms(boxes, scores, 0.6, 0.4)
			for i in range(len(pick)):
                                det_result.append([cls, scores[pick][i], boxes[pick][i]])

		img_show = plot_boxes_cv2(img_bgr, det_result, detections, count_in, count_out, names)
		cv2.imshow("Object Detection", img_show)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()

