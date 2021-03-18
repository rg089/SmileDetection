import imutils
from cvTools.ConvNets.LeNet import LeNet
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

args = imutils.get_args(dataset=True, save_model=True)
data = []
labels = []

for imagePath in sorted(list(imutils.list_images(args["dataset"]))):
    img = cv2.imread(imagePath, 0)
    img = imutils.resize(img, width=28)
    img = img_to_array(img)
    data.append(img)

    label = imagePath.split("\\")[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
X_train, X_test = imutils.normalize(X_train, X_test)
enc, y_train, y_test = imutils.encodeY(y_train, y_test, ohe=False)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Bringing balance to the dataset
classTotals = y_train.sum(axis=0)
total = classTotals.sum()
maxi = classTotals.max()
classWeights = {i: maxi/classTotals[i] for i in range(len(classTotals))}

model = LeNet.build(28, 28, 1, 2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] model compiled. commencing training....")

epochs = 17
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=classWeights, batch_size=64, epochs=epochs, verbose=1)

print("[INFO] evaluating model..")
predictions = model.predict(X_test, batch_size=64)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=enc.classes_))

print("[INFO] saving model...")
model.save(args["model"], save_format="h5")

print("[INFO] saving plot...")
plt = imutils.plot_model(history, epochs, title="Training Loss and Accuracy")
plt.savefig("plots\\training_data")



