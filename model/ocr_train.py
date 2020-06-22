# encoding: utf-8
import matplotlib.pylab as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
sys.path.append("../process")
import data_input
from train_network import Lenet

def train(aug, model, train_x, train_y, test_x, test_y):

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    model.fit(aug.flow(train_x,train_y,batch_size=batch_size), validation_data=(test_x,test_y), steps_per_epoch=len(train_x)//batch_size, epochs=epochs, verbose=1)
    model.save("../predict/ocr_model.h5")

if __name__ =="__main__":
    channel = 3
    height = 32
    width = 32
    class_num = 11
    norm_size = 32
    batch_size = 32
    epochs = 40
    model = Lenet.neural(channel=channel, height=height, width=width, classes=class_num)
    train_x, train_y = data_input.load_data("../data/ocr_train", norm_size, class_num)
    test_x, test_y = data_input.load_data("../data/ocr_val", norm_size, class_num)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    train(aug,model,train_x,train_y,test_x,test_y)





