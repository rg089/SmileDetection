import imutils
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


args = imutils.get_args(pretrained_model=True, single_image=True, use_video=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model(args["model"])


if args["image"]:
    img = cv2.imread(args["image"])
    height, width = img.shape[:2]
    if width > 750 or width < 150:
        img = imutils.resize(img, width=600)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_clone = img.copy()
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(rects) == 0:
        roi = cv2.resize(gray, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        notSmiling, smiling = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"
        cv2.putText(img_clone, f"{label}: {max(notSmiling, smiling) * 100}%", (width//4, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 25), 2)
    for fX, fY, fW, fH in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        notSmiling, smiling = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"
        textpos = (fX, fY - 10)
        if fY < 30:
            if fY + fH < height - 40:
                textpos = (fX, fY + fH + 20)
            elif fX > 50:
                textpos = (0, fY)
            else:
                textpos = (fX + fW + 10, fY)
        cv2.putText(img_clone, f"{label}: {max(notSmiling, smiling) * 100}%", textpos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 255, 15), 1)
        cv2.rectangle(img_clone, (fX, fY), (fX + fW, fY + fH), (20, 255, 15), 2)
    imutils.showImage(img_clone, "Prediction")


else:
    if args["video"]:
        cap = cv2.VideoCapture(args["video"])
    else:
        cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if args["video"] and not ret:
            break
        img = frame
        height, width = img.shape[:2]
        if width > 750 or width < 150:
            img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_clone = img.copy()

        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for fX, fY, fW, fH in rects:
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            notSmiling, smiling = model.predict(roi)[0]
            label = "Smiling" if smiling > notSmiling else "Not Smiling"
            textpos = (fX, fY-10)
            if fY < 30:
                if fY + fH < height - 40:
                    textpos = (fX, fY + fH + 20)
                elif fX > 50:
                    textpos = (0, fY)
                else:
                    textpos = (fX + fW + 10, fY)
            cv2.putText(img_clone, f"{label}: {max(notSmiling, smiling) * 100}%", textpos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 255, 15), 1)
            cv2.rectangle(img_clone, (fX, fY), (fX + fW, fY + fH), (20, 255, 15), 2)
        cv2.imshow("Detected Video", img_clone)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


