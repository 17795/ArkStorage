# encoding: utf8
from flask import *
import keras
from keras.preprocessing.image import img_to_array
import imutils.paths as paths
import cv2
import os
import numpy as np
import matplotlib.pylab as plt
import json
import base64


channel = 3
height = 32
width = 32
class_num = 62
norm_size = 32
batch_size = 32
epochs = 40

f = open("data/matlist.json", encoding='utf-8')
matlist = json.load(f)

inventory = {}

item_model = keras.models.load_model("predict/item_model.h5")
ocr_model = keras.models.load_model("predict/ocr_model.h5")

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"

@app.route('/storage', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        print('recevied data')
        # image_base64 = request.data.get("image")
        f = json.loads(request.data)
        image_base64 = f['image']
        missing_padding = 4 - len(image_base64) % 4
        if missing_padding:
            image_base64 += '=' * missing_padding
        imgData = base64.b64decode(image_base64)
        nparr = np.frombuffer(imgData, np.uint8)
        scene = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print(scene.shape)
        wh_ratio = scene.shape[1]/scene.shape[0]
        scene = cv2.resize(scene, (int(720*wh_ratio), 720))
        circles = cv2.HoughCircles(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=55, maxRadius=65)
        circles_int = np.uint16(np.around(circles))
        inventory = {}
        rois = []
        roi_copies = []
        for c in circles[0]:
            box_size = int(c[2] * 2.4)
            x = int(c[0]-box_size//2)
            y = int(c[1]-box_size//2)
            if y < 0 or x < 0 or y+box_size >= scene.shape[0] or x+box_size >= scene.shape[1]:
                continue
            roi = scene[y:y+box_size, x:x+box_size, :]
            roi = cv2.resize(roi, (128, 128))
            roi_copy = roi.copy()
            roi_copies.append(roi_copy)
            roi = roi / 255 * 2 - 1
            roi = np.transpose(roi, (2, 0, 1))
            rois.append(roi)
        rois = np.stack(rois, 0)
        for i in range(len(rois)):
            roi_copy = roi_copies[i]
            img1 = roi_copy
            img1 = cv2.resize(img1,(norm_size,norm_size))
            img1 = img_to_array(img1)/255.0
            img1 = np.expand_dims(img1,axis=0)
            result = item_model.predict(img1)
            proba = np.max(result)
            predict_label = np.argmax(result)
            chinese = ""
            src = ""
            for mat in matlist:
                if (str(matlist[mat]['label']) == str(predict_label)):
                    src = matlist[mat]['src']
                    chinese = mat
                    break
            number_area = roi_copy[85:115, 45:115, :]
            number_area_gray = cv2.cvtColor(number_area, cv2.COLOR_BGR2GRAY)
            threshed = cv2.adaptiveThreshold(number_area_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, -45)
            contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            valid_box = []
            for ct in contours:
                if cv2.contourArea(ct)>10:
                    x, y, w, h = cv2.boundingRect(ct)
                    if 10/45 < w/h  and w/h < 35 / 45 and 12<h and h<18 and abs(y+h//2 - 15) < 5:
                        valid_box.append((x, y, w, h))
            valid_box = sorted(valid_box, key=lambda x:x[0])
            valid_box_2 = []
            num_digits = 1
            for box in valid_box:
                x, y, w, h = box
                if len(valid_box_2)==0 or (abs(x - valid_box_2[-1][0]) > 7 and abs(x - valid_box_2[-1][0]) < 15):
                    valid_box_2.append(box)
            for j, box in enumerate(valid_box_2):
                x, y, w, h = box
            num = '0'
            for box in valid_box_2:
                x, y, w, h = box
                digit_image = threshed[y:y+h, x:x+w]
                digit_image = np.pad(digit_image, ((3,3),(3,3)), mode='constant')
                digit_image = cv2.resize(digit_image, (28, 28))
                img1 = cv2.resize(digit_image,(norm_size,norm_size))
                img1 = np.expand_dims(img1, axis=2)
                img1 = np.concatenate((img1, img1, img1), axis=-1)
                imgtemp = img1
                img1 = img_to_array(img1)/255.
                img1 = np.expand_dims(img1,axis=0)
                result = ocr_model.predict(img1)
                proba = np.max(result)
                predict_label = np.argmax(result)
                if (predict_label == 0):
                    cv2.imwrite(chinese+'.png', imgtemp)
                if str(predict_label) != '10':
                    num += str(predict_label)
            num = int(num)
            # print(chinese, num)
            if chinese in matlist:
                inventory[src] = num
        print(str(inventory))
        return json.dumps(inventory, ensure_ascii=False)
    return

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')