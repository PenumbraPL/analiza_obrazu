from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt


def debug(e_img, cnts):
	e_img = ~e_img
	e_img =  cv2.cvtColor(e_img, cv2.COLOR_GRAY2RGB)
	cv2.drawContours(e_img, cnts, -1, (0,0,255))
	plt.imshow(e_img)
	plt.show()

def round_to_multiple(number, multiple):
	return multiple * round(number / multiple)

def round_up(num, val, delta):
	if num < val+delta or num > val-delta:
		return True
	return False

def resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim)
    return resized

def ocr(filepath):
	print("Loading OCR model...")
	model = load_model("./models/model")

	image = cv2.imread(filepath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # bgr - imread load in this format
	blurred = cv2.GaussianBlur(gray, (3, 3), 3)
	cv2.addWeighted(gray, 2, blurred, -1, 0)
	kernel = np.ones((3, 3), np.uint8)
	edged = cv2.Canny(blurred, 30, 90)
	e_img = edged = cv2.dilate(edged, kernel, iterations=1)
	

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

	#debug(e_img, cnts)

	#image_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#image_neg =	cv2.bitwise_not(image_thresh)
	#print(image_neg.shape)
	#text_histogram = np.sum(image_neg, axis=1, keepdims=True) / 255
	#plt.plot(text_histogram)
	#plt.show()

	characters = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		roi = gray[y:y + h, x:x + w]
		threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(threshold_h, threshold_w) = threshold.shape

		if threshold_w > threshold_h:
			threshold = resize(threshold, width=32)
		else:
			threshold = resize(threshold, height=32)

		(threshold_h, threshold_w) = threshold.shape
		dX = int(max(0, 32 - threshold_w) / 2.0)
		dY = int(max(0, 32 - threshold_h) / 2.0)
		corrected = cv2.copyMakeBorder(threshold, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
		corrected = cv2.resize(corrected, (32, 32))

		corrected = corrected.astype("float32") / 255.0
		corrected = np.expand_dims(corrected, axis=-1)

		characters.append((corrected, (x, y, w, h)))

	boxes = [b[1] for b in characters]
	characters = np.array([c[0] for c in characters], dtype="float32")

	predictions = model.predict(characters)
	label_names = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
	letter_list = []

	for (prediction, (x, y, w, h)) in zip(predictions, boxes):
		i = np.argmax(prediction)
		probability = prediction[i]
		label = label_names[i]

		letter_list.append((x, y, label))
		print("{} - {:.2f}%".format(label, probability * 100))
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, label, (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	
	print(letter_list)
	letter_list.sort(key= lambda x : (round_to_multiple(x[1], 200), x[0]))
	print(letter_list)
	letter_list = list(map( lambda x: x[2], letter_list))
	print(letter_list)

	return letter_list, image