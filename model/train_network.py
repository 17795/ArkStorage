# encoding:utf8
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
import keras.backend as K

class Lenet: 

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":
            input_shape = (height,width,channel)
        model = Sequential()
        model.add(Conv2D(20,(5,5),padding="same",activation="relu",input_shape=input_shape,name="conv1"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool1"))
        model.add(Conv2D(50,(5,5),padding="same",activation="relu",name="conv2",))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool2"))
        model.add(Flatten())
        model.add(Dense(500,activation="relu",name="fc1"))
        model.add(Dense(classes,activation="softmax",name="fc2"))

        return model

