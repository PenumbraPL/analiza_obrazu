import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2
from helpers import *
from resnet import *


EPOCHS = 10
INIT_LR = 1e-1
BS = 128

print("Loading datasets...")
(az_data, az_labels) = load_az_dataset("./A_Z Handwritten Data.csv")
(digits_data, digits_labels) = load_mnist_dataset()

az_labels += 10
data = np.vstack([az_data, digits_data])
labels = np.hstack([az_labels, digits_labels])

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0

le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)
classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.20, stratify=labels, random_state=42)


aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

print("Compiling model...")

model = ResNet34(shape=(32,32,1), classes=len(le.classes_))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


print("Training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	class_weight=classWeight,
	verbose=1)

labelNames = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

print("Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

print("Saving network...")
model.save("./model", save_format="h5")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("./result.png")

images = []
for i in np.random.choice(np.arange(0, len(testY)), size=(36,)):
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]

	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)

	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)

	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
	images.append(image)

montage = build_montages(images, (96, 96), (7, 7))[0]
cv2.imshow("Results", montage)
cv2.waitKey(0)