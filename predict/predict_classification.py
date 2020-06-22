# encoding: utf-8
import keras
from keras.preprocessing.image import img_to_array
import imutils.paths as paths
import cv2
import os
import numpy as np
import matplotlib.pylab as plt


channel = 3
height = 32
width = 32
class_num = 62
norm_size = 32
batch_size = 32
epochs = 40

test_path = "../data/test"
image_paths = sorted(list(paths.list_images(test_path)))
model = keras.models.load_model("traffic_model.h5")
for each in image_paths:
    image = cv2.imread(each)
    image = cv2.resize(image,(norm_size,norm_size))
    image = img_to_array(image)/255.0
    image = np.expand_dims(image,axis=0)
    result = model.predict(image)
    proba = np.max(result)
    predict_label = np.argmax(result)
    print(each, predict_label)

