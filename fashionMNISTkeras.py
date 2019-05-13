import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
print(tf.VERSION)
print(tf.keras.__version__)

#defining our callback to stop when we get great accuracy
class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

#Load our Data into Two Groups, Training and Validation
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Reshape and normalize our images so that they can be fed into a DL Model
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

#Defining our Model in Keras is Easy! 
model = keras.models.Sequential([
    #1 Convolutional layer to extract data
  keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    #1 Pooling Layer to compress the data
  keras.layers.MaxPooling2D(2, 2),
    #Flatten data layer to make it fit
  keras.layers.Flatten(),
    #Then 1 Dense layer that is actually running our inference
  keras.layers.Dense(128, activation='relu'),
    #And then our final layer of outputs, 1 for each of our fashion categories
  keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
model.save('fashion_model.h5')
print("Done Training Model")
