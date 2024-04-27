# test.py
import cv2
import os
from keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
from pygame import mixer
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier("haar cascade files\haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("haar cascade files\haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("haar cascade files\haarcascade_righteye_2splits.xml")

model = load_model('model/model.h5')

def detect_drowsiness(frame):
    score = 0
    rpred = [99]
    lpred = [99]

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255
        r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
        predict_r = model.predict(r_eye)
        rpred = np.argmax(predict_r, axis=1)

        if rpred[0] == 0:
            return "Closed"

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255
        l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
        predict_l = model.predict(l_eye)
        lpred = np.argmax(predict_l, axis=1)

        if lpred[0] == 0:
            return "Closed"

    return "Open"

@app.route('/api/detect_drowsiness', methods=['POST'])
def api_detect_drowsiness():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    nparr = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    status = detect_drowsiness(frame)

    if status == "Closed":
        sound.play()

    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True)
